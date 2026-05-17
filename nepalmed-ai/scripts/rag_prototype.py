"""
NepalMed AI: Med-RAG Prototype
Uses Ollama (Local LLM) and ChromaDB (Local Vector Store)
"""

try:
    from langchain_ollama import OllamaLLM
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyMuPDFLoader
    import chromadb
except ImportError:
    print("⚠️ Missing dependencies. Run: pip install ollama chromadb langchain-ollama langchain-community pymupdf sentence-transformers")

class NepalMedRAG:
    def __init__(self, model_name="llama3.2", persist_directory="data/vectors"):
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.persist_directory = persist_directory
        self.vector_db = None

    def ingest_document(self, pdf_path):
        """Loads a PDF, chunks it, and adds to the vector store."""
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        if not self.vector_db:
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_db.add_documents(chunks)
        print(f"✅ Ingested: {pdf_path}")

    def query(self, question):
        """Retrieves context from the vector store and generates an answer with source attribution."""
        if not self.vector_db:
            return "No documents ingested yet."
        
        # Retrieval
        docs = self.vector_db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source", "Unknown Source") for doc in docs]))
        
        # Augmented Prompt
        prompt = f"""
        You are NepalMed AI, a clinical assistant. Use the following context from Nepal Medical Guidelines to answer the question.
        If the answer is not in the context, say you do not know.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        answer = self.llm.invoke(prompt)
        
        return {
            "answer": answer,
            "sources": sources
        }

if __name__ == "__main__":
    print("NepalMed AI: Med-RAG Prototype Initialized")
    # rag = NepalMedRAG()
    # print(rag.query("How to treat malaria in Nepal?"))
