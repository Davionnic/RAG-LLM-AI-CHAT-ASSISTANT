"""Streamlit chat UI for the RAG assistant.

Run: streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src import config
from src.ingest import SUPPORTED_EXTENSIONS, index_size, ingest_paths
from src.rag import answer

st.set_page_config(page_title="RAG Chat Assistant", page_icon=":books:", layout="wide")

if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": str, "sources": list}
    st.session_state.messages = []


def save_and_ingest(uploaded_files) -> int:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for uf in uploaded_files:
        dest = config.DATA_DIR / uf.name
        dest.write_bytes(uf.getbuffer())
        paths.append(dest)
    return ingest_paths(paths)


# --- Sidebar: document management ---
with st.sidebar:
    st.title("Documents")

    uploaded = st.file_uploader(
        "Upload documents",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest uploaded files", type="primary"):
        try:
            with st.spinner("Embedding and indexing..."):
                added = save_and_ingest(uploaded)
            st.success(f"Indexed {added} chunk(s) from {len(uploaded)} file(s).")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    st.divider()
    try:
        n_chunks = index_size()
    except Exception:
        n_chunks = 0
    st.metric("Indexed chunks", n_chunks)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Chat via Groq API. Embeddings run locally on first ingest."
    )

# --- Main: chat ---
st.title("RAG Chat Assistant")
st.caption(
    f"Ask questions about your documents. Powered by {config.CHAT_MODEL} "
    "(Groq) + Chroma."
)

if index_size() == 0:
    st.info("No documents indexed yet. Upload files in the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src_doc in msg["sources"]:
                    st.markdown(f"**{src_doc['source']}**")
                    st.text(src_doc["snippet"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        (m["role"], m["content"]) for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply, source_docs = answer(question, history)
        except Exception as e:
            reply, source_docs = f"Something went wrong: {e}", []

        st.markdown(reply)

        sources = []
        seen = set()
        for doc in source_docs:
            key = (doc.metadata.get("source", "unknown"), doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            snippet = doc.page_content[:400]
            if len(doc.page_content) > 400:
                snippet += "..."
            sources.append(
                {"source": doc.metadata.get("source", "unknown"), "snippet": snippet}
            )
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}**")
                    st.text(s["snippet"])

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
