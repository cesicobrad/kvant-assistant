import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Kvant Assistant",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0f0f0f;
    color: #e8e4dc;
}

section[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #2a2a2a;
}

section[data-testid="stSidebar"] * {
    color: #e8e4dc !important;
}

h1, h2, h3 {
    font-family: 'DM Mono', monospace !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em;
}

.kvant-header {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.25rem;
}

.kvant-title {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 300;
    color: #e8e4dc;
    margin: 0 0 2rem 0;
    line-height: 1.2;
}

.answer-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c8b88a;
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1rem 0;
    font-size: 15px;
    line-height: 1.7;
    color: #e8e4dc;
}

.source-chip {
    display: inline-block;
    background: #1f1f1f;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 3px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #888;
    margin: 3px 4px 3px 0;
}

.source-snippet {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #555;
    margin: 4px 0 12px 0;
    line-height: 1.5;
    padding-left: 8px;
    border-left: 1px solid #2a2a2a;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4a9e6b;
    display: inline-block;
    margin-right: 6px;
}

.meta-line {
    font-size: 11px;
    color: #444;
    font-family: 'DM Mono', monospace;
    margin-top: 0.5rem;
}

.suggested-btn {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 8px 14px;
    font-size: 13px;
    color: #888;
    cursor: pointer;
    margin: 4px 0;
    text-align: left;
    width: 100%;
    transition: all 0.15s;
}

.stTextInput input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput input:focus {
    border-color: #c8b88a !important;
    box-shadow: none !important;
}

.stButton button {
    background: #c8b88a !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
}

.stButton button:hover {
    background: #d4c49a !important;
}

div[data-testid="stSpinner"] {
    color: #666 !important;
}

.divider {
    border: none;
    border-top: 1px solid #1f1f1f;
    margin: 1.5rem 0;
}

.score-bar-bg {
    background: #1f1f1f;
    border-radius: 2px;
    height: 4px;
    margin-top: 4px;
}

.score-bar-fill {
    background: #c8b88a;
    border-radius: 2px;
    height: 4px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_rag():
    from app import build_rag
    return build_rag()

with st.sidebar:
    st.markdown('<p class="kvant-header">System</p>', unsafe_allow_html=True)

    with st.spinner("Loading knowledge base..."):
        try:
            vector_store, llm, prompt = load_rag()
            st.markdown('<span class="status-dot"></span><span style="font-size:12px;color:#4a9e6b;">Ready</span>', unsafe_allow_html=True)
            system_ok = True
        except Exception as e:
            st.error(f"Setup failed: {e}")
            st.info("Make sure you've run `python ingest.py` first and Ollama is running.")
            system_ok = False

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="kvant-header">Suggested queries</p>', unsafe_allow_html=True)

    suggestions = [
        "What vector database does Kvant use in production?",
        "What caused the NLB production outage?",
        "What LLM does the Mercator system use?",
        "Which projects use Qdrant on Hetzner?",
        "What is Kvant's break-even for self-hosting Llama?",
        "What is the street address of Kvant's office?",
    ]

    for s in suggestions:
        if st.button(s, key=f"sug_{s[:20]}", use_container_width=True):
            st.session_state.pending_query = s

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="kvant-header">Corpus</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#555;font-family:\'DM Mono\',monospace;">30 documents<br>Qdrant · Mistral · MiniLM</p>', unsafe_allow_html=True)

    if st.button("Re-ingest documents", use_container_width=True):
        import subprocess
        with st.spinner("Running ingest.py..."):
            result = subprocess.run(["python", "ingest.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Done. Reload the page.")
            else:
                st.error(result.stderr[-500:])

st.markdown('<p class="kvant-header">Kvant Consulting</p>', unsafe_allow_html=True)
st.markdown('<h1 class="kvant-title">Internal knowledge assistant</h1>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        default_val = st.session_state.pending_query
        query = st.text_input(
            "question",
            value=default_val,
            placeholder="Ask anything about Kvant's internal documents...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Ask →")

if submitted and query and system_ok:
    st.session_state.pending_query = ""
    with st.spinner("Searching..."):
        from app import answer_question
        answer, docs = answer_question(query, vector_store, llm, prompt, verbose=False)

    st.session_state.history.insert(0, {
        "query": query,
        "answer": answer,
        "docs": docs
    })

if st.session_state.pending_query and not submitted:
    st.rerun()

for i, item in enumerate(st.session_state.history):
    st.markdown(f'<p style="font-size:13px;color:#555;font-family:\'DM Mono\',monospace;margin-bottom:0.25rem;">Q — {item["query"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

    sources_seen = set()
    for doc in item["docs"]:
        src = doc.metadata.get("source", "?")
        chunk = doc.metadata.get("chunk_index", "?")
        key = f"{src}_{chunk}"
        if key not in sources_seen:
            sources_seen.add(key)
            snippet = doc.page_content[:180].replace("\n", " ").strip()
            st.markdown(f'<span class="source-chip">{src} · chunk {chunk}</span>', unsafe_allow_html=True)
            st.markdown(f'<p class="source-snippet">"{snippet}..."</p>', unsafe_allow_html=True)

    if i < len(st.session_state.history) - 1:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)