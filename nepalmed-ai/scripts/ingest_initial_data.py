"""
NepalMed AI: Initial Data Ingestion Script
Ingests the seed medical guidelines into the local vector store.
"""

import sys
import os

# Add the scripts directory to path if needed
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rag_prototype import NepalMedRAG

def main():
    print("🚀 Starting NepalMed AI Initial Ingestion...")
    
    # Initialize RAG
    rag = NepalMedRAG(persist_directory="data/vectors")
    
    # Path to data
    data_dir = "data/raw"
    files = [
        "nepal_antibiotic_guidelines_2014.txt",
        "nepal_stp_basic_health_2078.txt",
        "nepal_kala_azar_protocol.txt",
        "nepal_maternal_health_guidelines.txt"
    ]
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            print(f"📖 Ingesting {filename}...")
            # Simple text loader for txt files (PyMuPDF is for PDFs)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Since rag_prototype expects ingest_document(pdf_path), 
            # I'll add a helper for txt or manually chunk it here.
            # For now, let's just use the logic in rag_prototype 
            # but adapted for the prototype's current state.
            
            from langchain_core.documents import Document
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(content)
            docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
            
            if not rag.vector_db:
                rag.vector_db = Chroma.from_documents(
                    documents=docs,
                    embedding=rag.embeddings,
                    persist_directory=rag.persist_directory
                )
            else:
                rag.vector_db.add_documents(docs)
                
            print(f"✅ Successfully ingested {len(docs)} chunks from {filename}")
        else:
            print(f"⚠️ File not found: {file_path}")

    print("\n🎉 Initial Ingestion Complete!")
    print("You can now query the system using the rag_prototype.py or a custom script.")

if __name__ == "__main__":
    main()
