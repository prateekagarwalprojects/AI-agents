import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

st.set_page_config(page_title="RAGlocal UI", page_icon="R", layout="wide")
st.title("RAGlocal UI")
st.caption("Separate frontend for your local RAG backend")

backend_url_default = os.getenv("RAG_UI_BACKEND_URL", "http://127.0.0.1:8000")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Connection")
    backend_url = st.text_input("Backend URL", value=backend_url_default)

    st.subheader("Current .env")
    st.text_input("RAGLOCAL_PDF_DIR", value=os.getenv("RAGLOCAL_PDF_DIR", ""), disabled=True)
    st.text_input("RAGLOCAL_COLLECTION", value=os.getenv("RAGLOCAL_COLLECTION", "raglocal_docs"), disabled=True)
    st.text_input("RAGLOCAL_DB_DIR", value=os.getenv("RAGLOCAL_DB_DIR", "raglocal_db"), disabled=True)

    if st.button("Health Check"):
        try:
            r = requests.get(f"{backend_url}/health", timeout=30)
            r.raise_for_status()
            st.success("Backend reachable")
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Index PDFs")
    override_pdf_dir = st.text_input(
        "Optional PDF directory override",
        value="",
        placeholder="Leave empty to use RAGLOCAL_PDF_DIR from .env",
    )
    reset_index = st.checkbox("Reset collection before indexing", value=True)

    if st.button("Run Indexing", type="primary"):
        payload = {"reset": reset_index}
        if override_pdf_dir.strip():
            payload["pdf_dir"] = override_pdf_dir.strip()

        with st.spinner("Indexing in progress..."):
            try:
                r = requests.post(f"{backend_url}/index", json=payload, timeout=600)
                r.raise_for_status()
                data = r.json()
                st.success(data.get("message", "Indexing done"))
                st.json(data.get("summary", {}))
                logs = data.get("logs", [])
                if logs:
                    st.code("\n".join(logs), language="text")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.subheader("Collection Stats")
    if st.button("Refresh Stats"):
        try:
            r = requests.get(f"{backend_url}/stats", timeout=30)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(f"Stats fetch failed: {e}")

with col2:
    st.subheader("Ask Question")
    question = st.text_area("Question", height=120, placeholder="Type your question here...")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = {"question": question.strip(), "top_k": int(top_k)}
            with st.spinner("Retrieving answer..."):
                try:
                    r = requests.post(f"{backend_url}/ask", json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])

                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Citations")
                    if citations:
                        for c in citations:
                            st.write(f"- {c}")
                    else:
                        st.write("No citations returned.")

                    st.session_state.chat_history.append(
                        {"question": question.strip(), "answer": answer, "citations": citations}
                    )
                except Exception as e:
                    st.error(f"Ask failed: {e}")

st.subheader("Session Q&A History")
if st.button("Clear History"):
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.info("No questions asked yet.")
else:
    for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
        st.markdown(f"**Q{i}:** {item['question']}")
        st.write(item["answer"])
        if item.get("citations"):
            st.caption("Sources: " + "; ".join(item["citations"]))
        st.divider()
