import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.retriever import NepalGuidelinesRetriever

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Check if index exists
    if not os.path.exists("data/rag_index/nepal_index.faiss"):
        print("❌ RAG Index not found. Please run 'python scripts/build_nepal_index.py' first.")
        return

    retriever = NepalGuidelinesRetriever()
    
    # Test Queries
    queries = [
        "Pott's spine treatment guidelines",
        "lumbar canal stenosis classification",
        "antitubercular drugs for spine TB"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, k=2)
        
        if not results:
            print("No results found.")
            continue
            
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] Score: {res['score']:.4f}")
            print(f"Source: {res['metadata']['source']} (Page {res['metadata']['page']})")
            print(f"Text: {res['text'][:200]}...")

if __name__ == "__main__":
    main()
