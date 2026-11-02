# Kvant Assistant — Internal Knowledge Base RAG

A conversational assistant over a corpus of 30 synthetic internal
documents from a fictional Slovenian AI consultancy (Kvant Consulting).
Built as a take-home assignment for 5element.

## What it does

Answers natural-language questions about internal documents — meeting
notes, project proposals, technical reports, post-mortems, Slack
threads — returning answers with source citations and refusing to
answer when the information isn't in the corpus.

## The meta layer

The corpus is a fictional AI consulting company that builds RAG systems
for clients. Their internal documents discuss Qdrant, hallucination
problems, embedding models, and the exact architectural tradeoffs I
faced building this system. The assistant is literally answering
questions about itself — ask it "What vector database does Kvant use
in production?" or "What caused the NLB production outage?" to see
what I mean.

## Architecture

Classic RAG, fully local — no external APIs, no cloud, no ongoing costs:

- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` — handles Slovenian and English, runs on GPU
- **Vector store:** Qdrant (local embedded mode)
- **Retrieval:** hybrid: dense cosine similarity plus BM25 keyword search, merged with reciprocal rank fusion, top-5 chunks
- **LLM:** Qwen 2.5 14B via Ollama (local), temperature 0 so answers are repeatable
- **Eval:** LLM-as-judge, 15 questions across 4 categories — 14/15 on each of three runs

Architectural rationale and tradeoffs are in `REPORT.md`.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Qwen 2.5 14B pulled (about 9 GB; runs on a 12 GB GPU)

## Setup

**1. Install Ollama and pull the model:**
ollama pull qwen2.5:14b

**2. Clone the repo and create a virtual environment:**
git clone https://github.com/cesicobrad/kvant-assistant
cd kvant-assistant
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

**3. Install dependencies:**
pip install -r requirements.txt

**4. Build the vector store:**
python ingest.py

## Usage
python ingest.py        # build the vector store (run once, or after changing data/)
streamlit run ui.py     # browser UI — recommended
python app.py           # CLI chat
python eval.py          # run the evaluation suite

## Project structure
.
├── data/                  # 30 source documents (.pdf, .json, .md)
├── local_qdrant/          # Qdrant vector store (created by ingest.py)
├── ingest.py              # document loader, chunker, and indexer
├── app.py                 # RAG pipeline and CLI chat
├── ui.py                  # Streamlit browser interface
├── eval.py                # evaluation runner with LLM-as-judge
├── eval_questions.json    # 15 evaluation questions
├── eval_results.json      # output of the last eval run
├── requirements.txt
├── REPORT.md
└── README.md

## Evaluation results

| Category      | Score     |
|---------------|-----------|
| Simple        | 7/7       |
| Multi-hop     | 4/4       |
| Contradiction | 1/2       |
| Unanswerable  | 2/2       |
| **Overall**   | **14/15** |

Same score on three separate runs, with the same single miss. Q12 asks
which LLM the Mercator Polica system uses. The answer is in a table in
`doc_003.pdf`, but the passage that holds it doesn't make the top five,
so the assistant says it doesn't know rather than guessing. I left it
failing instead of tuning settings to this one question.

### Update, November 2026

The first version (Mistral 7B, dense search only) reported 14/15, but
re-running it gave 14, 11 and 12: the model answered slightly
differently each time, and it also graded itself. Three changes made
the result stable:

- JSON documents are indexed from their `content` field instead of as
  raw JSON, so their text is chunked properly.
- Hybrid retrieval: BM25 keyword search catches exact names and figures
  ("Mercator Polica", "Phase 1") that embeddings blur together.
- Qwen 2.5 14B at temperature 0 for answering and judging.

With Mistral 7B, the same retrieval changes scored 11–13, so the bigger
model is what made the difference. Set `KVANT_MODEL=mistral` to
compare on smaller hardware.

## Suggested questions to try
What vector database does Kvant use in production?
What caused the NLB production outage in September 2024?
What LLM does the Mercator Polica system use?
Which projects use Qdrant on Hetzner infrastructure?
What is Kvant's break-even for self-hosting Llama 3.1 70B?
What is the street address of Kvant's main office?
