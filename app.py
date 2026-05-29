"""
Streamlit UI — Document Q&A System (RAG)
Run with: streamlit run app.py
"""

import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="DocuAsk — Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Main font & background */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Chat messages */
    .user-message {
        background: #1e293b;
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
    }
    .assistant-message {
        background: #0f2027;
        border-left: 3px solid #10b981;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.95rem;
    }

    /* Source chunks */
    .source-chunk {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 4px 0;
    }

    /* Status badges */
    .badge-success {
        background: #064e3b;
        color: #34d399;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-pending {
        background: #1e1b4b;
        color: #818cf8;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #1e3a5f;
    }
    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 6px 0;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.9rem;
    }

    /* Input box */
    .stTextInput input {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://127.0.0.1:8000"

# Session state 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_ingested" not in st.session_state:
    st.session_state.pdf_ingested = False
if "ingest_info" not in st.session_state:
    st.session_state.ingest_info = {}

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at aistudio.google.com",
    )

    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=2, max_value=8, value=4,
        help="More chunks = more context but slower response",
    )

    st.markdown("---")
    st.markdown("### Upload Document PDF")

    uploaded_file = st.file_uploader(
        "Upload any PDF document",
        type=["pdf"],
        help="PDF will be split, embedded, and stored in ChromaDB",
    )

    if uploaded_file and not st.session_state.pdf_ingested:
        if st.button("Ingest PDF", use_container_width=True):
            with st.spinner("Embedding document into vector store..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.pdf_ingested = True
                        st.session_state.ingest_info = data
                        st.success(f"Ingested! {data['pages_processed']} pages → {data['chunks_created']} chunks")
                    else:
                        st.error(f"Ingestion failed: {resp.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to FastAPI backend. Make sure `uvicorn main:app --reload` is running.")

    if st.session_state.pdf_ingested:
        info = st.session_state.ingest_info
        st.markdown(f"""
        <div style='background:#064e3b;padding:12px;border-radius:8px;margin-top:8px'>
            <div style='color:#34d399;font-weight:600;font-size:0.85rem'>Document Loaded</div>
            <div style='color:#6ee7b7;font-size:0.8rem;margin-top:4px'>
                {info.get('pages_processed','?')} pages &nbsp;|&nbsp; 
                {info.get('chunks_created','?')} chunks
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Re-ingest new PDF", use_container_width=True):
            st.session_state.pdf_ingested = False
            st.session_state.ingest_info = {}
            st.rerun()

    st.markdown("---")
    st.markdown("### Stack")
    st.markdown("""
    - **LangChain** (LCEL)  
    - **Gemini 2.5 Flash**  
    - **ChromaDB** vector store  
    - **all-MiniLM-L6-v2** embeddings  
    - **FastAPI** backend  
    """)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>DocuAsk — Document Q&A</h1>
    <p>Upload any PDF and ask questions — answers are grounded in your document, not hallucinated.</p>
</div>
""", unsafe_allow_html=True)

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    status = "Document loaded" if st.session_state.pdf_ingested else "No document loaded"
    color = "#34d399" if st.session_state.pdf_ingested else "#818cf8"
    st.markdown(f"<span style='color:{color};font-size:0.85rem;font-weight:600'>{status}</span>", unsafe_allow_html=True)
with col2:
    key_status = "API key set" if gemini_key else "No API key"
    color = "#34d399" if gemini_key else "#818cf8"
    st.markdown(f"<span style='color:{color};font-size:0.85rem;font-weight:600'>{key_status}</span>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<span style='color:#94a3b8;font-size:0.85rem'>{len(st.session_state.chat_history)} messages</span>", unsafe_allow_html=True)

st.markdown("---")

# ── Suggested questions ───────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("#### Try asking...")
    suggestions = [
        "What is this document about?",
        "Summarize the key findings.",
        "What are the main conclusions?",
        "What methodology was used?",
        "What are the limitations mentioned?",
        "What future work is suggested?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"<div class='user-message'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"View {len(msg['sources'])} source chunks retrieved"):
                for j, src in enumerate(msg["sources"]):
                    st.markdown(f"<div class='source-chunk'><strong>Chunk {j+1}:</strong><br>{src}</div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        "Ask a question",
        placeholder="e.g. What environment was used for training?",
        label_visibility="collapsed",
        key="user_input",
    )

with col_btn:
    ask_clicked = st.button("Ask →", use_container_width=True, type="primary")

# ── Handle question ───────────────────────────────────────────────────────────
question = None
if ask_clicked and user_input.strip():
    question = user_input.strip()
elif st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    # Triggered by suggestion button
    question = st.session_state.chat_history[-1]["content"]
    st.session_state.chat_history.pop()  # will be re-added below

if question:
    if not gemini_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
    elif not st.session_state.pdf_ingested:
        st.warning("Please upload and ingest your PDF first using the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Retrieving chunks and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={
                        "question": question,
                        "gemini_api_key": gemini_key,
                        "top_k": top_k,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources_used"],
                    })
                else:
                    err = resp.json().get("detail", "Unknown error")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {err}",
                        "sources": [],
                    })
            except requests.exceptions.ConnectionError:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Cannot reach FastAPI backend. Is `uvicorn main:app --reload` running?",
                    "sources": [],
                })

        st.rerun()
