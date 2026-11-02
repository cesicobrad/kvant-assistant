import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore as Qdrant
    from qdrant_client import QdrantClient
    try:
        from langchain_ollama import OllamaLLM as Ollama
    except ImportError:
        from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"[!] Missing package: {e}")
    sys.exit(1)

COLLECTION_NAME = "kvant_knowledge_base"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = os.environ.get("KVANT_MODEL", "qwen2.5:14b")
TEMPERATURE = float(os.environ.get("KVANT_TEMPERATURE", "0"))  # 0: the same question gets the same answer
TOP_K = 6
CANDIDATES = 20  # per retriever, before fusion
RRF_K = 60
MAX_PER_SOURCE = 2  # keep room for other documents; questions often need more than one

SYSTEM_PROMPT = """You are an internal AI assistant for Kvant Consulting. Answer ONLY using the provided context below.

STRICT RULES:
1. If the information is not explicitly stated in the context, you MUST respond with exactly: "I don't know based on the internal documents."
2. Never guess, infer, or use outside knowledge. Only state what is directly in the context.
3. Cite the source number for each fact (e.g. "According to Source 2, ...").
4. If sources contradict each other, present both views and flag the contradiction.
5. Be concise.

Context:
{context}"""

# Common English and Slovenian words that say nothing about which document matches.
STOPWORDS = set("""a an and are as at be by did do does for from has have how in is it its of on or that the this to
was were what when where which who why will with je in na za da se so ali kaj kdo kje kako""".split())

def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]

class HybridRetriever:
    """Dense (embeddings) + keyword (BM25) search, merged with reciprocal rank fusion.

    Dense search finds passages with similar meaning; BM25 catches exact names and
    figures ("Mercator Polica", "Phase 1") that embeddings can blur together.
    """

    def __init__(self, client, vector_store):
        self.vector_store = vector_store
        self.docs = []
        offset = None
        while True:
            points, offset = client.scroll(COLLECTION_NAME, limit=256, offset=offset, with_payload=True, with_vectors=False)
            for p in points:
                payload = p.payload or {}
                self.docs.append(Document(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {}) or {}))
            if offset is None:
                break
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

    @staticmethod
    def _key(doc):
        return (doc.metadata.get("source"), doc.metadata.get("chunk_index"))

    def search(self, query, k=TOP_K):
        dense = self.vector_store.similarity_search(query, k=CANDIDATES)
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
        keyword = [self.docs[i] for i in top if scores[i] > 0]
        fused, by_key = {}, {}
        for ranking in (dense, keyword):
            for rank, doc in enumerate(ranking):
                key = self._key(doc)
                by_key.setdefault(key, doc)
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        picked, per_source = [], {}
        for key in sorted(fused, key=fused.get, reverse=True):
            if per_source.get(key[0], 0) >= MAX_PER_SOURCE:
                continue
            per_source[key[0]] = per_source.get(key[0], 0) + 1
            picked.append(by_key[key])
            if len(picked) == k:
                break
        return picked

def build_rag():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    client = QdrantClient(path="./local_qdrant")
    vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    retriever = HybridRetriever(client, vector_store)
    llm = Ollama(model=LLM_MODEL, temperature=TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return retriever, llm, prompt

def answer_question(question, vector_store, llm, prompt, k=TOP_K, verbose=True):
    docs = vector_store.search(question, k=k)
    context_text = ""
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "?")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        context_text += f"-- Source {i+1} [{src}, chunk {chunk_idx}] --\n{doc.page_content}\n\n"
    formatted = prompt.format(context=context_text, input=question)
    answer = llm.invoke(formatted)
    if verbose:
        print(f"\nKvant AI: {answer}\n")
        print("--- Sources Used ---")
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "?")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            snippet = doc.page_content[:140].replace("\n", " ")
            print(f"  Source {i+1}: {src} (chunk {chunk_idx})")
            print(f"    \"{snippet}...\"")
        print("--------------------")
    return answer, docs

def main():
    print("Loading Kvant AI Assistant...")
    try:
        vector_store, llm, prompt = build_rag()
    except Exception as e:
        print(f"Setup error: {e}\nDid you run ingest.py? Is Ollama running?")
        return

    print("\n=== KVANT CONSULTING AI ASSISTANT ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            print("Goodbye.")
            break
        print("\nThinking...")
        try:
            answer_question(user_input, vector_store, llm, prompt)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()