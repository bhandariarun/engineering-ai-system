import os
import requests
import streamlit as st

st.set_page_config(page_title="Engineering AI Assistant", page_icon="AI", layout="wide")

st.title("Engineering AI Assistant")
st.caption("Retrieval-grounded answers with tool calling and graceful offline fallback")
api_url = st.sidebar.text_input("Backend URL", os.getenv("API_URL", "http://localhost:8000"))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

if question := st.chat_input("Ask about the assistant architecture..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            response = requests.post(f"{api_url.rstrip('/')}/chat", json={"question": question}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            meta = f"Provider: {payload['provider']}"
            if payload.get("cached"):
                meta += " | cached"
            if payload.get("degraded"):
                meta += " | degraded mode"
            st.markdown(payload["answer"])
            if payload.get("citations"):
                st.caption(meta + " | Sources: " + ", ".join(item["source"] for item in payload["citations"]))
            else:
                st.caption(meta)
            st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "meta": meta})
        except requests.RequestException as error:
            st.error(f"Backend unavailable: {error}")
