"""Streamlit chat UI for the CS2 RAG assistant.

    streamlit run app/ui.py

Talks to the FastAPI service (API_URL), shows the grounded answer, and lists
the cited source passages in an expandable panel with their retrieval scores.
"""
import os
import re

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CS2 RAG Assistant", page_icon="🎯")
st.title("🎯 CS2 RAG Assistant")
st.caption("Grounded answers about Counter-Strike 2, with sources cited inline.")

with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API URL", API_URL)
    top_n = st.slider("Passages to retrieve", 3, 8, 5)
    st.markdown(
        "Ask about weapons, the economy, map callouts, utility, round rules, "
        "or strategy. The assistant answers only from its curated knowledge "
        "base and cites the passages it used."
    )

if "history" not in st.session_state:
    st.session_state.history = []


def esc(text):
    # keep dollar amounts ($4750) from being rendered as LaTeX math
    return text.replace("$", "\\$")


def render_sources(sources, answer=""):
    cited = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    with st.expander(f"Sources ({len(sources)} passages)", expanded=False):
        for s in sources:
            mark = "✅ cited" if s["n"] in cited else ""
            head = f"**[{s['n']}] {s['source']}"
            if s["section"] and s["section"] != "overview":
                head += f" › {s['section']}"
            head += f"** {mark}"
            st.markdown(head)
            scores = []
            if s.get("rerank_score") is not None:
                scores.append(f"rerank {s['rerank_score']:.2f}")
            if s.get("dense_score") is not None:
                scores.append(f"dense {s['dense_score']:.2f}")
            if s.get("bm25_score") is not None:
                scores.append(f"bm25 {s['bm25_score']:.2f}")
            if scores:
                st.caption(" · ".join(scores))
            body = esc(s["body"][:600])
            st.markdown("\n".join("> " + ln for ln in body.splitlines()))
            st.divider()


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(esc(turn["content"]))
        if turn.get("sources"):
            render_sources(turn["sources"], turn["content"])

if prompt := st.chat_input("Ask a CS2 question…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(esc(prompt))
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering…"):
            try:
                r = requests.post(f"{api_url}/chat",
                                  json={"question": prompt, "top_n": top_n},
                                  timeout=180)
                r.raise_for_status()
                data = r.json()
                answer, sources = data["answer"], data["sources"]
            except Exception as e:
                answer, sources = f"Request failed: {e}", []
        st.markdown(esc(answer))
        if sources:
            render_sources(sources, answer)
    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources})
