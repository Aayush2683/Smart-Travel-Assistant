import os
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

import re, textwrap, pickle
from pathlib import Path
import faiss, torch
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
torch.set_num_threads(1); torch.set_num_interop_threads(1)

BASE       = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE / "Data" / "RAG Data" / "Processed"
INDEX_PATH = DATA_DIR / "travel_docs_faiss.index"
META_PATH  = DATA_DIR / "travel_docs_meta.pkl"
LLAMA_GGUF = BASE / "RAG" / "Llama-3.2-1B-Instruct-f16.gguf"

EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
INIT_K       = 20            # initial FAISS neighbours
MAX_K        = 100           # max widening
PHRASE_BONUS = 0.25           # exact bigram/tri-gram match
KEEP_CHUNKS  = 3             # chunks fed to Llama
MAX_TOKENS   = 256

STOPWORDS = {
    "the","is","a","an","to","for","of","and","in","on","that","this",
    "with","as","by","at","from","it","its","be","are","was","were","or"
}

DISPLAY = {
    "Travelite India – Brochure": "Travelite India Brochure",
    "Air India – Security Regulations":  "Air India Security Regulations (2024)",
    "Air India Express – Fees & Charges":     "Air India Express Fees & Charges (2023)",
    "IndiGo – ZED Travel Policy":         "IndiGo ZED Policy (2023)",
    "Alliance Air – Baggage Policy":   "Alliance Air Baggage Policy (2024)",
}

print("🔹 Loading FAISS index & models …")
index = faiss.read_index(str(INDEX_PATH))
meta  = pickle.load(open(META_PATH, "rb"))
embed = SentenceTransformer(EMB_MODEL, device="cpu")
llm   = Llama(model_path=str(LLAMA_GGUF),
              n_ctx=4096,
              n_threads=os.cpu_count())

def phrases(text: str):
    toks = [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]
    uni  = {" ".join(toks[i:i+n]) for n in (2,3) for i in range(len(toks)-n+1)}
    return uni

def get_hits(q_vec, k):
    D, I = index.search(q_vec, k)
    return [{**meta[i], "sim": float(D[0][r])} for r, i in enumerate(I[0])]

def rescored(query: str):
    q_vec = embed.encode([query], normalize_embeddings=True).astype("float32")
    q_ph  = phrases(query)
    k     = INIT_K
    while True:
        hits = get_hits(q_vec, k)
        for h in hits:
            ph_bonus = PHRASE_BONUS if any(p in h["text"].lower() for p in q_ph) else 0
            h["score"] = h["sim"] + ph_bonus
        hits.sort(key=lambda x: x["score"], reverse=True)
        if hits[0]["score"] > 0.10 or k >= MAX_K:
            return hits
        k = min(k*2, MAX_K)

def build_prompt(question, chunks):
    context = "\n\n---\n".join(chunks)
    return (
        "Answer the USER question **CORRECTLY** using ONLY the information provided in CONTEXT. "
        "Give Short and Consice to the point answer and it should be 100percent correct for the given question"
        "Reply in NO MORE THAN 50 words. "
        "if there is any price being involved then focus on using the correct currency"
        "If the answer is not present, respond exactly with 'information not found'.\n\n"
        "**REMEMBER TO KEEP THE ANSWER SHORT AND CONSICE**"
        f"CONTEXT:\n{context}\n\nUSER: {question}\nASSISTANT:"
    )

def answer(q: str) -> str:

    hits = rescored(q)[:KEEP_CHUNKS]
    prompt = build_prompt(q, [h["text"] for h in hits])
    resp   = llm.create_completion(prompt=prompt,
                                   max_tokens=MAX_TOKENS,
                                   temperature=0.0,
                                   stop=["\n\n"])
    ans = resp["choices"][0]["text"].strip()

    title = DISPLAY.get(hits[0]["doc"], hits[0]["doc"])
    cite  = f"{title} (chunk {hits[0]['chunk_id']})"
    return f"According to {title}, {ans} \n\n*Source – {cite}*"

if __name__ == "__main__":
    while True:
        query = input("\nAsk your travel question (or 'quit'): ").strip()
        if query.lower() in {"quit","exit"}:
            break
        answer(query)