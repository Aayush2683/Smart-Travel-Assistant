import os
os.environ.update(
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1"
)

import re, pickle, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pdfplumber
import faiss
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

BASE      = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE / "Data" / "RAG Data" / "Processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = DATA_DIR / "travel_docs_faiss.index"
META_PATH  = DATA_DIR / "travel_docs_meta.pkl"

DOCS: Dict[str, Dict[str, str]] = {
    "Travelite India – Brochure": Path("/Users/aayush2683/Projects/Solvus AI Intern/Data/RAG Data/RAW/45.pdf"),
    "Air India – Security Regulations": Path("/Users/aayush2683/Projects/Solvus AI Intern/Data/RAG Data/RAW/security-regulations-dangerous-goods-restricted-items.pdf"),
    "Air India Express – Fees & Charges": Path("/Users/aayush2683/Projects/Solvus AI Intern/Data/RAG Data/RAW/AIX-FeesandCharges-12-OCT-23.pdf"),
    "IndiGo – ZED Travel Policy": Path("/Users/aayush2683/Projects/Solvus AI Intern/Data/RAG Data/RAW/ZEDPolicy.pdf"),
    "Alliance Air – Baggage Policy": Path("/Users/aayush2683/Projects/Solvus AI Intern/Data/RAG Data/RAW/baggage-policy.pdf"),
}

CHUNK_MIN, CHUNK_MAX = 500, 800
EMBED_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text(pdf: Path) -> str:
    with pdfplumber.open(str(pdf)) as doc:
        return "\n".join(page.extract_text() or "" for page in doc.pages)

def clean(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
    return txt.strip()

def chunk(txt: str, mn=CHUNK_MIN, mx=CHUNK_MAX) -> List[str]:
    words, buf, chunks = txt.split(), [], []
    for w in words:
        buf.append(w)
        if len(" ".join(buf)) >= mx:
            chunks.append(" ".join(buf)); buf = []
    if buf and len(" ".join(buf)) >= mn:
        chunks.append(" ".join(buf))
    return chunks

def build_index():
    print("🔹 Loading embedder …")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    all_vecs, metadata = [], []

    for name, pdf_path in DOCS.items():
        if not pdf_path.exists():
            print(f"{pdf_path} not found — skipping.")
            continue

        print(f"⇢  Processing {pdf_path.name}")
        text = clean(extract_text(pdf_path))
        chunks = chunk(text)
        if not chunks:
            print("No extractable text — skipped.")
            continue

        vecs = model.encode(chunks,
                            batch_size=32,
                            normalize_embeddings=True,
                            show_progress_bar=False)

        all_vecs.append(vecs)
        metadata.extend(
            {"doc": name, "chunk_id": i, "text": c}
            for i, c in enumerate(chunks)
        )
        print(f"   - {len(chunks)} chunks embedded")

    if not all_vecs:
        raise RuntimeError("No documents processed — nothing to index!")

    vec_array = np.vstack(all_vecs).astype("float32")
    print(f"\n🔧 Building FAISS index ({vec_array.shape[0]} vectors)…")
    index = faiss.IndexFlatIP(vec_array.shape[1])   # cosine (vectors L2-normed)
    index.add(vec_array)

    print("Saving artefacts →", DATA_DIR)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Finished!")
    print(" •", INDEX_PATH)
    print(" •", META_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    if (INDEX_PATH.exists() and META_PATH.exists()) and not args.rebuild:
        print("Index already exists — use --rebuild to regenerate.")
    else:
        build_index()