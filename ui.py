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

.eval-pass {
    background: #0d1f15;
    border: 1px solid #1a3d28;
    border-left: 3px solid #4a9e6b;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}

.eval-fail {
    background: #1f0d0d;
    border: 1px solid #3d1a1a;
    border-left: 3px solid #9e4a4a;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}

.eval-question {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #666;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.eval-answer {
    font-size: 13px;
    color: #e8e4dc;
    line-height: 1.6;
}

.eval-score {
    font-family: 'DM Mono', monospace;
    font-size: 32px;
    font-weight: 300;
    color: #c8b88a;
    margin: 1rem 0;
}

.divider {
    border: none;
    border-top: 1px solid #1f1f1f;
    margin: 1.5rem 0;
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

.category-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 8px;
    background: #1f1f1f;
    color: #666;
    border: 1px solid #2a2a2a;
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
            st.session_state.active_tab = "chat"

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

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

with tab_chat:
    st.markdown('<p class="kvant-header">Kvant Consulting</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="kvant-title">Internal knowledge assistant</h1>', unsafe_allow_html=True)

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

with tab_eval:
    st.markdown('<p class="kvant-header">Kvant Consulting</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="kvant-title">Evaluation suite</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:14px;color:#555;margin-bottom:2rem;">15 questions across 4 categories. LLM-as-judge scoring.</p>', unsafe_allow_html=True)

    if st.button("Run all 15 questions →", use_container_width=False):
        import json
        from pathlib import Path
        from app import answer_question

        questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))

        JUDGE_PROMPT = """You are evaluating a RAG system answer. Be generous — partial credit counts.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {actual}
CATEGORY: {category}

Scoring:
- unanswerable: correct=true if the answer says it doesn't know or can't find it. correct=false if it invents facts.
- contradiction: correct=true if it mentions both values OR correctly identifies the deployed/final value.
- simple/multi_hop: correct=true if the answer contains the main fact from expected, even if incomplete or differently worded.

Respond with ONLY this JSON on one line:
{{"correct": true, "reasoning": "one sentence"}}
"""

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, q in enumerate(questions):
            status.markdown(f'<p style="font-size:12px;color:#666;font-family:\'DM Mono\',monospace;">Running Q{q["id"]} — {q["question"][:60]}...</p>', unsafe_allow_html=True)
            progress.progress((i + 1) / len(questions))

            try:
                actual, docs = answer_question(q["question"], vector_store, llm, prompt, verbose=False)
            except Exception as e:
                actual = f"ERROR: {e}"
                docs = []

            judge_input = JUDGE_PROMPT.format(
                question=q["question"],
                expected=q["expected"],
                actual=actual,
                category=q["category"],
            )

            try:
                judgment_raw = llm.invoke(judge_input)
                import re, json as j
                match = re.search(r'\{[^{}]*"correct"\s*:\s*(true|false)[^{}]*\}', judgment_raw, re.DOTALL)
                if match:
                    judgment = j.loads(match.group(0))
                else:
                    judgment = {"correct": False, "reasoning": judgment_raw[:120]}
            except Exception as e:
                judgment = {"correct": False, "reasoning": str(e)}

            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "actual": actual,
                "correct": judgment.get("correct", False),
                "reasoning": judgment.get("reasoning", ""),
            })

        status.empty()
        progress.empty()
        st.session_state.eval_results = results

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        correct_count = sum(r["correct"] for r in results)
        total = len(results)

        st.markdown(f'<p class="eval-score">{correct_count}/{total}</p>', unsafe_allow_html=True)

        by_cat = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r["correct"])

        cols = st.columns(len(by_cat))
        for col, (cat, vals) in zip(cols, by_cat.items()):
            with col:
                st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:10px;color:#666;text-transform:uppercase;letter-spacing:0.1em;">{cat}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:20px;color:#c8b88a;">{sum(vals)}/{len(vals)}</p>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        for r in results:
            css_class = "eval-pass" if r["correct"] else "eval-fail"
            mark = "✓" if r["correct"] else "✗"
            st.markdown(f'''
                <div class="{css_class}">
                    <p class="eval-question"><span class="category-badge">{r["category"]}</span>Q{r["id"]} — {r["question"]}</p>
                    <p class="eval-answer">{mark} {r["reasoning"]}</p>
                    <p style="font-family:\'DM Mono\',monospace;font-size:11px;color:#444;margin-top:0.5rem;border-top:1px solid #2a2a2a;padding-top:0.5rem;">{r["actual"][:200]}...</p>
                </div>
            ''', unsafe_allow_html=True)