import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

def main():
    # Added a visual marker so you know EXACTLY which version is running
    print("\n--- RUNNING CUSTOM LOADER V4 ---")
    print("Starting data ingestion...")

    data_dir = Path('./data')
    if not data_dir.exists() or not data_dir.is_dir():
        print("Error: The 'data' folder does not exist in the current directory.")
        return

    documents = []
    print(f"Scanning for files in {data_dir.absolute()}...")
    
    # Manually read files to completely bypass LangChain's broken file loaders
    for filepath in data_dir.rglob('*.*'):
        if not filepath.is_file():
            continue
            
        try:
            # Try standard UTF-8
            text = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to ignore errors if there are weird characters
            text = filepath.read_text(encoding='utf-8', errors='ignore')
            
        # Manually create the Document object
        documents.append(Document(page_content=text, metadata={"source": str(filepath)}))
        
    print(f"Successfully loaded {len(documents)} documents manually.")

    if len(documents) == 0:
        print("Wait! I didn't find any documents. Make sure they are inside the 'data' folder.")
        return

    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks of text.")

    print("Creating local embeddings (this will use your RTX 5070!)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Initializing local Qdrant database to bypass LangChain bugs...")
    
    # 1. Initialize the Qdrant client directly
    client = QdrantClient(path="./local_qdrant")
    
    # 2. Manually create the collection. 
    # The paraphrase-multilingual-MiniLM-L12-v2 model outputs vectors of exactly 384 dimensions.
    client.recreate_collection(
        collection_name="kvant_knowledge_base",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    # 3. Connect the LangChain wrapper to our manually created collection
    vector_store = Qdrant(
        client=client,
        collection_name="kvant_knowledge_base",
        embeddings=embeddings
    )

    print("Saving documents into Qdrant...")
    vector_store.add_documents(chunks)

    print("Success! Data ingestion is complete. Your AI's library is ready.")

if __name__ == "__main__":
    main()