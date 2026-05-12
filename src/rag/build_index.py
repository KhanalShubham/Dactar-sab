import os
import json
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

logger = logging.getLogger("IndexBuilder")

class IndexBuilder:
    """Handles text chunking, embedding, and FAISS index construction."""
    
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def create_chunks(self, pages_content, source_name, chunk_size=512, overlap_sentences=2):
        """
        Splits pages into overlapping chunks.
        chunk_size is approximate character count or word count? 
        Spec says ~512 tokens (~300-400 words).
        """
        all_text = ""
        # Flatten pages but keep track of page numbers if possible
        # For simplicity in this version, we'll chunk the whole doc text
        # but a better way is to track page per chunk.
        
        chunks = []
        
        for page in pages_content:
            text = page['text']
            sentences = sent_tokenize(text)
            
            current_chunk = []
            current_count = 0
            
            for i, sent in enumerate(sentences):
                sent_words = sent.split()
                current_chunk.append(sent)
                current_count += len(sent_words)
                
                if current_count >= 350: # Target ~350 words (~512 tokens)
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": source_name,
                            "page": page['page_no'],
                            "char_count": len(chunk_text)
                        }
                    })
                    
                    # Create overlap: keep last N sentences
                    current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
                    current_count = sum(len(s.split()) for s in current_chunk)
            
            # Add remaining sentences as a final chunk for the page
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": source_name,
                        "page": page['page_no'],
                        "char_count": len(chunk_text)
                    }
                })
        
        return chunks

    def build_faiss_index(self, chunks, output_dir="data/rag_index"):
        """Embeds chunks and saves FAISS index."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        texts = [c['text'] for c in chunks]
        metadata = [c['metadata'] for c in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks...")
        # Normalize embeddings for Cosine Similarity (IP on normalized vectors)
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save everything
        index_path = os.path.join(output_dir, "nepal_index.faiss")
        chunks_path = os.path.join(output_dir, "chunks.pkl")
        meta_path = os.path.join(output_dir, "metadata.json")
        config_path = os.path.join(output_dir, "index_config.json")
        
        faiss.write_index(index, index_path)
        
        with open(chunks_path, 'wb') as f:
            pickle.dump(texts, f)
            
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        with open(config_path, 'w') as f:
            json.dump({
                "model_name": self.model_name,
                "dimension": dimension,
                "total_chunks": len(texts),
                "created_at": str(np.datetime64('now'))
            }, f, indent=2)
            
        logger.info(f"Index and {len(texts)} chunks saved to {output_dir}")
        return index_path
