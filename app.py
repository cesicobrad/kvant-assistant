import sys
import warnings
warnings.filterwarnings("ignore")

# Try to use the most up-to-date imports, but fall back if needed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    # Try the newer Ollama import first
    try:
        from langchain_ollama import OllamaLLM as Ollama
    except ImportError:
        from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    print(f"\n[!] Missing a required package: {e}")
    print("[!] To fix this, run: pip install langchain langchain-core langchain-huggingface qdrant-client")
    sys.exit(1)

def main():
    print("Loading your Kvant AI Assistant...")

    # 1. Load the same free local embeddings you used for ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 2. Connect to the Qdrant database we built earlier
    try:
        client = QdrantClient(path="./local_qdrant")
        vector_store = Qdrant(
            client=client,
            collection_name="kvant_knowledge_base",
            embeddings=embeddings
        )
    except Exception as e:
        print(f"Error: Could not connect to the database. {e}")
        print("Did you run ingest.py first?")
        return

    # 3. Set up the FREE local LLM using Ollama
    # Ensure Ollama is running and you have run 'ollama run llama3.2'
    llm = Ollama(model="llama3.2")

    # 4. Give the AI its personality and rules (Prompt Engineering!)
    system_prompt = (
        "You are an internal AI assistant for Kvant Consulting. "
        "Use the provided context from internal documents to answer the user's question. "
        "If the answer is not in the context, explicitly say 'I don't know based on the internal documents.' "
        "Always respond professionally. "
        "\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    print("\n======================================")
    print(" === KVANT CONSULTING AI ASSISTANT ===")
    print("======================================")
    print("Type 'exit' to quit.\n")

    # 5. Start the chat loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down assistant. Goodbye!")
            break
            
        print("\nAI is thinking... (running locally on your GPU)\n")
        
        try:
            # RAG STEP 1: Find relevant documents using similarity_search
            # This is safer than using retriever.invoke() or client.search() directly
            docs = vector_store.similarity_search(user_input, k=3)
            
            # RAG STEP 2: Combine the text from the documents
            context_text = ""
            for i, doc in enumerate(docs):
                context_text += f"-- Source {i+1} --\n{doc.page_content}\n\n"
            
            # RAG STEP 3: Format the prompt
            formatted_prompt = prompt.format(context=context_text, input=user_input)
            
            # RAG STEP 4: Send to Ollama
            answer = llm.invoke(formatted_prompt)
            
            # Print the answer
            print(f"Kvant AI: {answer}")
            
            # Print the sources
            print("\n--- Sources Used ---")
            sources = set([doc.metadata.get('source', 'Unknown Document') for doc in docs])
            for source in sources:
                print(f"- {source}")
            print("--------------------\n")
            
        except Exception as e:
            print(f"\nError: Could not generate response. Details: {e}")

if __name__ == "__main__":
    main()