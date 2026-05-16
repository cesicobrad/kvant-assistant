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
except ImportError as e:
    print(f"[!] Missing package: {e}")
    sys.exit(1)

COLLECTION_NAME = "kvant_knowledge_base"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "mistral"
TOP_K = 5

SYSTEM_PROMPT = """You are an internal AI assistant for Kvant Consulting. Answer ONLY using the provided context below.

STRICT RULES:
1. If the information is not explicitly stated in the context, you MUST respond with exactly: "I don't know based on the internal documents."
2. Never guess, infer, or use outside knowledge. Only state what is directly in the context.
3. Cite the source number for each fact (e.g. "According to Source 2, ...").
4. If sources contradict each other, present both views and flag the contradiction.
5. Be concise.

Context:
{context}"""

def build_rag():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    client = QdrantClient(path="./local_qdrant")
    vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    llm = Ollama(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return vector_store, llm, prompt

def answer_question(question, vector_store, llm, prompt, k=TOP_K, verbose=True):
    docs = vector_store.similarity_search(question, k=k)
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