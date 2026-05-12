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
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", cache_dir="./data/embeddings_cache"):
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
            
            # MULTIMODAL KNOWLEDGE BASE
            self.MODALITIES = {
                "SPINE_MRI": "sagittal t2-weighted lumbar spine mri showing vertebral bodies and discs",
                "CHEST_XRAY": "frontal chest x-ray showing lungs, heart and ribs",
                "EXTREMITY_XRAY": "x-ray of human limb, leg, knee, or arm showing bones and joints"
            }
            
            self.TAGS_BY_MODALITY = {
                "SPINE_MRI": [
                    "normal lumbar lordosis", "grade I anterolisthesis", "degenerative spondylolisthesis",
                    "normal vertebral body marrow", "conus medullaris ends at L1-L2", "vertebral body hemangioma",
                    "Modic type I endplate changes", "Modic type II endplate changes", "vertebral compression fracture",
                    "normal thecal sac patency", "lateral recess narrowing", "moderate spinal canal stenosis",
                    "severe spinal canal stenosis", "cauda equina crowding", "patent neural foramina",
                    "foraminal stenosis contacting exiting nerve root", "normal disc height and signal",
                    "mild disc desiccation", "posterior disc bulge", "central disc protrusion",
                    "paracentral disc extrusion", "facet joint hypertrophy"
                ],
                "CHEST_XRAY": [
                    "normal clear lungs and pleura", "pneumonia with consolidation", "pleural effusion",
                    "cardiomegaly with enlarged heart shadow", "pneumothorax", "hilar lymphadenopathy",
                    "pulmonary nodule", "interstitial lung disease", "atelectasis", "rib fracture",
                    "hiatal hernia", "congestive heart failure"
                ],
                "EXTREMITY_XRAY": [
                    "normal bone alignment and joint space", "acute cortical fracture", "comminuted fracture",
                    "osteoarthritis with joint space narrowing", "marginal osteophytes", "joint dislocation",
                    "soft tissue swelling", "bone cyst", "osteopenia", "periosteal reaction"
                ]
            }
            
            logger.info("BiomedCLIP engine initialized with multimodal precision.")
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

    def detect_modality(self, image_bytes):
        """
        Zero-shot modality classifier to detect if image is Spine, Chest, or Extremity.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.preprocess_val(image).unsqueeze(0).to(self.device)
        
        modality_keys = list(self.MODALITIES.keys())
        modality_descriptions = list(self.MODALITIES.values())
        
        text_tokens = self.tokenizer(modality_descriptions).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()[0]
            
        best_idx = np.argmax(probs)
        detected = modality_keys[best_idx]
        logger.info(f"Detected modality: {detected} (confidence: {probs[best_idx]:.2f})")
        return detected

    def auto_tag_findings(self, image_bytes, candidate_labels=None, threshold=0.08):
        """
        Performs Precision Zero-Shot Tagging. 
        If candidate_labels is None, it auto-detects modality and uses appropriate labels.
        """
        # 1. Modality detection if labels not provided
        modality = "SPINE_MRI"
        if candidate_labels is None:
            modality = self.detect_modality(image_bytes)
            candidate_labels = self.TAGS_BY_MODALITY.get(modality, self.TAGS_BY_MODALITY["SPINE_MRI"])

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
                    results.append({"label": label, "score": float(prob), "modality": modality})
            
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

    def analyze_volume_consensus(self, images_list, threshold=0.10):
        """
        Runs precision tagging across a list of images (3D stack).
        Implements Phase 7: Slice Agreement Validation.
        """
        all_tags = []
        for img_bytes in images_list:
            tags = self.auto_tag_findings(img_bytes, threshold=threshold)
            all_tags.append(tags)
            
        consensus_map = {}
        for slice_tags in all_tags:
            for tag in slice_tags:
                label = tag['label']
                score = tag['score']
                if label not in consensus_map:
                    consensus_map[label] = []
                consensus_map[label].append(score)
        
        # SLICE AGREEMENT VALIDATION (Requirement: Findings must persist across 20% of slices)
        min_occurrences = max(1, len(images_list) // 5)
        final_consensus = []
        for label, scores in consensus_map.items():
            if len(scores) >= min_occurrences:
                avg_score = sum(scores) / len(images_list)
                final_consensus.append({"label": label, "score": avg_score})
        
        return sorted(final_consensus, key=lambda x: x['score'], reverse=True)

    def get_sir_map(self, findings):
        """
        Generates a 'Structured Intermediate Representation' (SIR).
        Phase 8: Expert-Tier Reasoning.
        """
        sir = {
            "L1-L2": {"disc": "Normal", "canal": "Patent", "foramina": "Patent", "facets": "Normal", "lateral_recess": "Patent"},
            "L2-L3": {"disc": "Normal", "canal": "Patent", "foramina": "Patent", "facets": "Normal", "lateral_recess": "Patent"},
            "L3-L4": {"disc": "Normal", "canal": "Patent", "foramina": "Patent", "facets": "Normal", "lateral_recess": "Patent"},
            "L4-L5": {"disc": "Normal", "canal": "Patent", "foramina": "Patent", "facets": "Normal", "lateral_recess": "Patent"},
            "L5-S1": {"disc": "Normal", "canal": "Patent", "foramina": "Patent", "facets": "Normal", "lateral_recess": "Patent"},
            "CONUS": "Normal termination at L1-L2",
            "ALIGNMENT": "Preserved lumbar lordosis",
            "BONE_MARROW": "Normal signal"
        }
        
        for f in findings:
            label = f['label'].lower()
            level = "L4-L5" # Default if not specified
            for l in sir.keys():
                if l.lower() in label: level = l
            
            if "disc" in label or "bulge" in label or "protrusion" in label:
                sir[level]["disc"] = f['label']
            if "canal" in label or "stenosis" in label:
                sir[level]["canal"] = f['label']
            if "foramina" in label:
                sir[level]["foramina"] = f['label']
            if "facet" in label:
                sir[level]["facets"] = f['label']
            if "recess" in label:
                sir[level]["lateral_recess"] = f['label']
                
        return sir
