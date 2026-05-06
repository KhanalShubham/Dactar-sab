import torch
import torch.nn.functional as F
from PIL import Image
import io
import os
import numpy as np
import logging
import faiss
import pickle
import open_clip
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MedicalCLIP")

class MedicalCLIPAnalyzer:
    """
    State-of-the-art Medical AI engine using BiomedCLIP (Microsoft).
    Provides high-precision zero-shot alignment for spinal MRI.
    """
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", cache_dir="./embeddings_cache"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.index_path = os.path.join(cache_dir, "medical_cases.index")
        self.meta_path = os.path.join(cache_dir, "medical_cases.pkl")
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        try:
            logger.info(f"Loading BiomedCLIP model: {model_name} on {self.device}")
            # Loading via open_clip for BiomedCLIP specialized architecture
            self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            logger.info("BiomedCLIP engine initialized with full medical precision.")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load BiomedCLIP engine: {e}")
            raise RuntimeError(f"Precision AI engine failed to load. Ensure open_clip_torch is installed and HF is reachable. Error: {e}")
            
        # Initialize Vector DB
        self.index = None
        self.metadata = []
        self._init_vector_db()

    def _init_vector_db(self):
        """Loads the FAISS index and metadata if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded Vector DB with {len(self.metadata)} cases.")
            except Exception as e:
                logger.error(f"Error loading Vector DB: {e}")
                self.index = None
        else:
            logger.info("Vector DB empty. Precision search will be available after indexing.")

    def get_image_embedding(self, image_bytes):
        """Extracts visual embedding vector using BiomedCLIP ViT encoder."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.preprocess_val(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # L2 Normalize for cosine similarity precision
            image_features = F.normalize(image_features, p=2, dim=-1)
            
        return image_features.cpu().numpy()

    def auto_tag_findings(self, image_bytes, candidate_labels=None, threshold=0.08):
        """
        Performs Precision Zero-Shot Tagging on the MRI using BiomedCLIP.
        """
        if candidate_labels is None:
            candidate_labels = [
                # VERTEBRAL ALIGNMENT
                "normal lumbar lordosis and vertebral alignment",
                "grade I anterolisthesis",
                "degenerative spondylolisthesis",
                "retrolisthesis of the vertebral body",
                
                # BONE MARROW & TECHNIQUE
                "T1 and T2 weighted sagittal sequences",
                "normal vertebral body bone marrow signal",
                "hemangioma of the vertebral body",
                "Modic type I endplate inflammatory changes",
                "Modic type II endplate fatty changes",
                "vertebral body compression fracture",
                "Schmorl's nodes in the vertebral endplates",

                # DISC ASSESSMENT
                "severe spinal canal stenosis with thecal sac compression",
                "central canal narrowing with nerve root crowding",
                "disc protrusion causing neural foraminal narrowing",
                "disc extrusion with migrated fragment",
                "T2 desiccation of the intervertebral disc",
                "annular fissure or high intensity zone",
                "healthy intervertebral disc signal and height",

                # FACETS & SOFT TISSUE
                "facet joint hypertrophy and arthropathy",
                "ligamentum flavum thickening",
                "cauda equina nerve root impingement",
                "paraspinal muscle atrophy or fatty infiltration"
            ]

        try:
            start_time = datetime.now()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_tensor = self.preprocess_val(image).unsqueeze(0).to(self.device)
            
            # Tokenize labels
            text_tokens = self.tokenizer(candidate_labels).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize for precision alignment
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                logits_per_image = 100.0 * image_features @ text_features.T
                probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Precision tagging completed in {duration:.3f}s")

            results = []
            for label, prob in zip(candidate_labels, probs):
                if prob >= threshold:
                    results.append({"label": label, "score": float(prob)})
            
            return sorted(results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return [] # No mock fallback allowed in Precision mode

    def find_similar_cases(self, query_image_bytes, top_k=5):
        """
        Semantic search for similar historical MRI cases based on visual embedding.
        """
        if self.index is None or len(self.metadata) == 0:
            logger.warning("Vector DB empty. Real-time search requires indexed data.")
            return []

        query_emb = self.get_image_embedding(query_image_bytes)
        D, I = self.index.search(query_emb.astype('float32'), top_k)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "path": self.metadata[idx]["path"],
                    "similarity": float(dist)
                })
        
        return results

    def validate_report(self, report_text, visual_tags):
        """
        AI Validator: Checks if the radiologist's text aligns with BiomedCLIP findings.
        """
        text_lower = report_text.lower()
        tag_labels = [t['label'].lower() for t in visual_tags if t['score'] > 0.1]
        
        found = []
        missing = []
        
        for label in tag_labels:
            # Semantic keyword overlap
            keywords = label.split()
            if any(kw in text_lower for kw in keywords if len(kw) > 3):
                found.append(label)
            else:
                missing.append(label)
        
        score = len(found) / len(tag_labels) if tag_labels else 1.0
        
        return {
            "score": score,
            "missing_findings": missing,
            "status": "VALID" if score > 0.8 else "WARNING"
        }
