from __future__ import annotations
import re, datetime as dt
from pathlib import Path
import streamlit as st
from dateutil.parser import parse as dt_parse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Prediction import inference
from RAG import rag_query

DATE_RE = r"\b\d{1,2}\s+[a-z]{3,9}\b|\b[a-z]{3,9}\s+\d{1,2}\b"
def route_query(txt: str) -> str:
    t = re.sub(r"\s+", " ", txt.lower().strip())
    is_price  = re.search(r"\b(predict|price|fare|cost)\b", t)
    has_route = re.search(r"\bfrom .+? to .+? on .+", t)
    has_date  = re.search(DATE_RE, t)
    return "ML" if (is_price and has_route and has_date) else "RAG"

st.set_page_config(page_title="Travel Assistant",
                   page_icon="✈️", layout="wide")

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap"
          rel="stylesheet">
    <style>
        html, body, textarea, input, button {
            font-family: 'Poppins', sans-serif !important;
            font-size: 1.35rem !important; color:#f7f7f7;
        }
        /* Neon grid background + noise */
        body::before{
            content:"";position:fixed;inset:0;z-index:-3;
            background:radial-gradient(circle at 50% 15%,rgba(0,255,200,0.09),transparent 60%),
                       repeating-linear-gradient(45deg,rgba(255,255,255,0.03) 0 1px,transparent 1px 80px),
                       linear-gradient(135deg,#181818 0%, #0f0f0f 100%);
        }
        body::after{
            content:"";position:fixed;inset:0;z-index:-2;
            background:url("https://www.transparenttextures.com/patterns/asfalt-light.png");
            opacity:0.15;
        }
        /* Chat bubbles */
        .stChatMessage{
            background:rgba(255,255,255,0.07);
            border:1px solid rgba(255,255,255,0.14);
            border-radius:18px;padding:1.3rem 1.5rem;margin-bottom:1.1rem;
            box-shadow:0 8px 22px rgba(0,0,0,0.45);
            backdrop-filter:blur(16px) saturate(120%);
        }
        .stChatMessage:nth-child(odd){
            background:rgba(0,180,255,0.12);
            border-color:rgba(0,180,255,0.30);
        }
        /* Markdown tweaks */
        .stMarkdown p{line-height:1.6;margin:0.45rem 0;}
        .stMarkdown code{color:#0ff;}
        /* full-width input */
        section[aria-label="chat input"] > div{width:100%;}
        textarea.stTextInput,input.stTextInput{
            font-size:1.2rem;padding:0.75rem 1.1rem;border-radius:14px;
            border:1px solid #0ff4;background:rgba(255,255,255,0.08);
            transition:box-shadow .25s ease;border-left-width:3px;
        }
        textarea.stTextInput:focus,input.stTextInput:focus{
            outline:none;border-color:#0ff;box-shadow:0 0 12px 2px rgba(0,255,255,0.55);
        }
        /* spinner */
        .stSpinner > div > div{border-top-color:#0ff !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 **Smart Travel Assistant**")
st.subheader("By Aayush Sharma")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)
        if m.get("shap_img"):
            st.image(m["shap_img"], caption="Feature impact (SHAP)")

if prompt := st.chat_input("Ask me anything about flights or travel…"):
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if route_query(prompt) == "ML":
                reply = inference.predict(prompt)
                st.markdown(reply, unsafe_allow_html=True)
                st.image("shap_plot.png", caption="Feature impact (SHAP)")
                st.session_state.messages.append(
                    {"role":"assistant","content":reply,"shap_img":"shap_plot.png"}
                )
            else:
                reply = rag_query.answer(prompt)
                st.markdown(reply, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role":"assistant","content":reply}
                )
    st.experimental_rerun()