import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pypdf import PdfReader
import psutil

COLLECTION_NAME = "kvant_knowledge_base"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384

def main():
    print("\n--- KVANT INGEST V5 ---")

    # Memory guard — inspired by the NLB post-mortem in the corpus,
    # which documents a production outage caused by reindexing without
    # checking available memory first.
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    safe_threshold_gb = 4.0
    print(f"Available RAM: {available_gb:.1f} GB")
    if available_gb < safe_threshold_gb:
        print(f"\n⚠ WARNING: Less than {safe_threshold_gb}GB RAM available.")
        print("The NLB post-mortem documents an outage caused by reindexing with insufficient memory.")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted. Free up memory and try again.")
            return

    data_dir = Path("./data")
    if not data_dir.exists():
        print("Error: ./data folder not found.")
        return

    documents = []
    print(f"Scanning {data_dir.absolute()}...")
    for filepath in sorted(data_dir.rglob("*.*")):
        if not filepath.is_file():
            continue

        if filepath.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(filepath))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                print(f"  Skipping {filepath.name}: {e}")
                continue
        else:
            try:
                text = filepath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = filepath.read_text(encoding="utf-8", errors="ignore")

        if not text.strip():
            continue

        documents.append(Document(
            page_content=text,
            metadata={"source": filepath.name}
        ))

    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents found in ./data")
        return

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.split_documents(documents)
    counts = {}
    for c in chunks:
        src = c.metadata["source"]
        counts[src] = counts.get(src, 0) + 1
        c.metadata["chunk_index"] = counts[src]
    print(f"Created {len(chunks)} chunks.")

    print("Loading embeddings (GPU if available)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Initializing Qdrant...")
    client = QdrantClient(path="./local_qdrant")

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    print("Indexing chunks...")
    vector_store.add_documents(chunks)
    print(f"Done. {len(chunks)} chunks indexed in '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()