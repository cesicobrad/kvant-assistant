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
- **Retrieval:** hybrid: dense cosine similarity plus BM25 keyword search, merged with reciprocal rank fusion; top 6 chunks, at most 2 per document; every chunk carries its document's title
- **LLM:** Qwen 2.5 14B via Ollama (local), temperature 0 so answers are repeatable
- **Eval:** LLM-as-judge, 15 questions across 4 categories — 15/15 on each of three runs, plus 11/11 on a second set written after the fixes

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
python eval.py heldout_questions.json heldout_results.json   # second question set

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
├── heldout_questions.json # 11 questions written after the fixes (python eval.py heldout_questions.json heldout_results.json)
├── heldout_results.json
├── requirements.txt
├── REPORT.md
└── README.md

## Evaluation results

| Category      | Score     |
|---------------|-----------|
| Simple        | 7/7       |
| Multi-hop     | 4/4       |
| Contradiction | 2/2       |
| Unanswerable  | 2/2       |
| **Overall**   | **15/15** |

Same score on three separate runs (temperature 0, so answers repeat).

**Second question set.** The retrieval changes below were made while
looking at this suite's failures, so I wrote 11 new questions afterwards
and ran them once without changing anything: **11/11**, including a
question in Slovenian and two the documents can't answer. They are
mostly single-document lookups, so they are easier than the original
multi-hop questions. Files: `heldout_questions.json`, `heldout_results.json`.

### Update, November 2026: how it got from 14/15 to 15/15

The first version (Mistral 7B, dense search only) reported 14/15, but
re-running it gave 14, 11 and 12: the model answered slightly
differently each time, and it also graded itself. What changed:

- **Duplicate chunks.** In local mode, `delete_collection` left old points
  in Qdrant's storage file, so each re-ingest added another copy of every
  chunk (804 points for 195 chunks). Search results were mostly copies.
  Ingest now rebuilds the index folder and uses stable chunk IDs, and it
  checks the stored count.
- **JSON documents** are indexed from their `content` field instead of as
  raw JSON.
- **Hybrid retrieval:** BM25 keyword search (English and Slovenian
  stopwords removed) merged with dense search, 20 candidates each.
- **Document titles on chunks.** A chunk from the middle of a report
  rarely names the project. The Q12 answer sat in a configuration table
  that never says "Mercator", so neither search linked it to the question.
- **At most 2 chunks per document** in the final 6, so one long document
  can't crowd out the others.
- **Qwen 2.5 14B at temperature 0** for answering and judging. Mistral 7B
  with the earlier retrieval changes scored 11–13; set
  `KVANT_MODEL=mistral` to compare on smaller hardware.

## Suggested questions to try
What vector database does Kvant use in production?
What caused the NLB production outage in September 2024?
What LLM does the Mercator Polica system use?
Which projects use Qdrant on Hetzner infrastructure?
What is Kvant's break-even for self-hosting Llama 3.1 70B?
What is the street address of Kvant's main office?
