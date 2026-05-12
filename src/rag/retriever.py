import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("Retriever")

class NepalGuidelinesRetriever:
    """Handles semantic search against the Nepal Medical Guidelines index."""
    
    def __init__(self, index_dir="data/rag_index"):
        self.index_dir = index_dir
        self.index = None
        self.chunks = []
        self.metadata = []
        self.model = None
        
        self._load_resources()

    def _load_resources(self):
        """Loads FAISS index, chunks, and metadata."""
        try:
            index_path = os.path.join(self.index_dir, "nepal_index.faiss")
            chunks_path = os.path.join(self.index_dir, "chunks.pkl")
            meta_path = os.path.join(self.index_dir, "metadata.json")
            config_path = os.path.join(self.index_dir, "index_config.json")
            
            if not all(os.path.exists(p) for p in [index_path, chunks_path, meta_path, config_path]):
                logger.warning("RAG Index components missing. Search will be unavailable.")
                return

            # Load Config to get model name
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loading embedding model: {config['model_name']}")
            self.model = SentenceTransformer(config['model_name'])
            
            logger.info("Loading FAISS index...")
            self.index = faiss.read_index(index_path)
            
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
                
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
                
            logger.info(f"Retriever ready with {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to load RAG resources: {e}")

    def retrieve(self, query, k=3):
        """
        Performs semantic search.
        Returns a list of (text, metadata) tuples.
        """
        if self.index is None or self.model is None:
            return []
            
        try:
            # Embed query (normalized for IP search)
            query_vector = self.model.encode([query], normalize_embeddings=True).astype('float32')
            
            # Search FAISS
            D, I = self.index.search(query_vector, k)
            
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx != -1 and idx < len(self.chunks):
                    results.append({
                        "text": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": float(dist)
                    })
            return results
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def format_for_prompt(self, results):
        """Formats results for injection into LLM system prompt."""
        if not results:
            return "No specific local guidelines found for this clinical context."
            
        formatted = "NEPAL MEDICAL GUIDELINES (Relevant Excerpts):\n"
        for i, res in enumerate(results, 1):
            source = res['metadata']['source']
            page = res['metadata']['page']
            formatted += f"[{i}] SOURCE: {source}, PAGE: {page}\n"
            formatted += f"TEXT: {res['text']}\n\n"
        return formatted
