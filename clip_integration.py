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
                "retrolisthesis",
                
                # BONE MARROW & CONUS
                "normal vertebral body bone marrow signal",
                "conus medullaris ends at L1-L2 with normal signal",
                "vertebral body hemangioma",
                "Modic type I endplate changes",
                "Modic type II endplate changes",
                "vertebral compression fracture",

                # SPINAL CANAL & THECAL SAC (Anatomically Correct)
                "normal thecal sac and canal patency",
                "mild thecal sac effacement",
                "lateral recess narrowing",
                "moderate lateral recess stenosis",
                "moderate spinal canal stenosis with cauda equina crowding",
                "severe spinal canal stenosis with nerve root crowding",
                "redundant nerve roots of the cauda equina",
                "congenital narrow spinal canal",
                
                # FORAMINA & ROOTS
                "patent neural foramina",
                "mild neural foraminal narrowing",
                "moderate foraminal stenosis contacting the exiting nerve root",
                "severe foraminal stenosis compressing the exiting nerve root",
                "exiting L5 nerve root contact",
                "traversing S1 nerve root contact",
                "traversing nerve root displacement",
                
                # DISCS
                "normal disc height and signal",
                "mild disc desiccation and height loss",
                "broad-based posterior disc bulge",
                "central disc protrusion",
                "paracentral disc extrusion",
                "foraminal disc protrusion",
                "annular fissure",
                
                # FACETS & SOFT TISSUE
                "facet joint hypertrophy and arthropathy",
                "ligamentum flavum thickening"
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

class SymbolicCausalValidator:
    """Enforces clinical logic: Findings must have a causal mechanism."""
    
    @staticmethod
    def validate_and_refine(sir):
        levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
        for level in levels:
            data = sir[level]
            canal = data["canal"].lower()
            
            # RULE: Severe Stenosis REQUIRES a cause (Bulge, Facets, Flavum, or Congenital)
            if "severe" in canal:
                cause_found = (
                    "bulge" in data["disc"].lower() or 
                    "protrusion" in data["disc"].lower() or
                    "extrusion" in data["disc"].lower() or
                    "hypertrophy" in data["facets"].lower() or
                    "thickening" in data["facets"].lower() or
                    "narrow" in canal # congenital
                )
                if not cause_found:
                    # Downgrade if no cause is identified to maintain medical integrity
                    sir[level]["canal"] = "Normal thecal sac and canal patency (AI overcall suppressed)"
                    sir[level]["_flag"] = "Contradiction: Stenosis removed due to lack of causal mechanism."
            
            # RULE: Terminology refinement
            if "age-related" in canal:
                sir[level]["canal"] = canal.replace("age-related", "degenerative")
                
        return sir

class CausalReasoningEngine:
    """Explains the 'Why' behind findings (Pathophysiological Reasoning)."""
    
    @staticmethod
    def derive_causality(findings):
        causality_map = {}
        labels = [f['label'].lower() for f in findings]
        
        # Level-by-level causality
        levels = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
        for level in levels:
            reasons = []
            if any(level.lower() in l and "stenosis" in l for l in labels):
                if any(level.lower() in l and "facet" in l for l in labels):
                    reasons.append("facet joint hypertrophy")
                if any(level.lower() in l and "ligamentum" in l for l in labels):
                    reasons.append("ligamentum flavum thickening")
                if any(level.lower() in l and "disc" in l for l in labels):
                    reasons.append("disc bulge/protrusion")
            
            if reasons:
                causality_map[level] = f"Stenosis is likely multifactorial, secondary to {', '.join(reasons)}."
            else:
                causality_map[level] = "Normal age-related findings or primary congenital narrowing."
                
        return causality_map

class AnatomicalConstraintEngine:
    """Enforces strict neuroanatomical logic rules to prevent AI hallucinations."""
    
    @staticmethod
    def apply_constraints(findings, levels=["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]):
        corrected = []
        for f in findings:
            label = f['label'].lower()
            score = f['score']
            
            # RULE 1: Physical Boundary Enforcement
            if "spinal cord" in label:
                f['label'] = f['label'].replace("spinal cord", "thecal sac/cauda equina")
            
            # RULE 2: EXPERT ROOT MAPPING (Phase 8)
            # Foraminal Stenosis -> Exiting Root (N)
            # Lateral Recess Stenosis -> Traversing Root (N+1)
            if "foramina" in label or "foraminal" in label:
                if "l3-l4" in label: f['root_info'] = "Exiting L3 nerve root"
                elif "l4-l5" in label: f['root_info'] = "Exiting L4 nerve root"
                elif "l5-s1" in label: f['root_info'] = "Exiting L5 nerve root"
            
            if "recess" in label:
                if "l3-l4" in label: f['root_info'] = "Traversing L4 nerve root"
                elif "l4-l5" in label: f['root_info'] = "Traversing L5 nerve root"
                elif "l5-s1" in label: f['root_info'] = "Traversing S1 nerve root"
            
            # RULE 3: Language Softening (Phase 8)
            # Replace 'compression' with safer terms unless extremely high confidence
            if "compression" in label and score < 0.8:
                f['label'] = f['label'].replace("compression", "mass effect/crowding")
            
            # RULE 4: Modifier Assignment
            if score > 0.85: f['modifier'] = "Definite"
            elif score > 0.65: f['modifier'] = "Likely"
            elif score > 0.40: f['modifier'] = "Probable"
            else: f['modifier'] = "Possible"
            
            corrected.append(f)
        return corrected

class ConsistencyEngine:
    """Detects and flags logical contradictions in findings."""
    
    @staticmethod
    def detect_contradictions(findings):
        labels = [f['label'].lower() for f in findings]
        conflicts = []
        
        # Conflict: Normal Canal + Severe Stenosis
        if any("normal" in l and "canal" in l for l in labels) and \
           any("severe" in l and "stenosis" in l for l in labels):
            conflicts.append("Contradiction: Normal canal findings co-exist with severe stenosis markers.")
            
        # Conflict: Normal Signal + Severe Desiccation
        if any("normal" in l and "disc" in l for l in labels) and \
           any("desiccation" in l or "height loss" in l for l in labels):
            conflicts.append("Contradiction: Normal disc signal co-exists with degenerative markers.")
            
        return conflicts
