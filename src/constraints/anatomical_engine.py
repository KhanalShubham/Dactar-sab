import torch

class AnatomicalConstraintEngine:
    """Enforces strict neuroanatomical logic rules to prevent AI hallucinations."""
    
    @staticmethod
    def apply_constraints(findings, levels=["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]):
        corrected = []
        for f in findings:
            label = f['label'].lower()
            score = f['score']
            modality = f.get('modality', 'SPINE_MRI')
            
            # Skip spine-specific constraints if not a spine study
            if modality != 'SPINE_MRI':
                # Basic rules for all modalities
                if score > 0.85: f['modifier'] = "Definite"
                elif score > 0.65: f['modifier'] = "Likely"
                elif score > 0.40: f['modifier'] = "Probable"
                else: f['modifier'] = "Possible"
                corrected.append(f)
                continue

            # RULE 1: Physical Boundary Enforcement
            if "spinal cord" in label:
                f['label'] = f['label'].replace("spinal cord", "thecal sac/cauda equina")
            
            # RULE 2: EXPERT ROOT MAPPING
            if "foramina" in label or "foraminal" in label:
                if "l3-l4" in label: f['root_info'] = "Exiting L3 nerve root"
                elif "l4-l5" in label: f['root_info'] = "Exiting L4 nerve root"
                elif "l5-s1" in label: f['root_info'] = "Exiting L5 nerve root"
            
            if "recess" in label:
                if "l3-l4" in label: f['root_info'] = "Traversing L4 nerve root"
                elif "l4-l5" in label: f['root_info'] = "Traversing L5 nerve root"
                elif "l5-s1" in label: f['root_info'] = "Traversing S1 nerve root"
            
            # RULE 3: Language Softening
            if "compression" in label and score < 0.8:
                f['label'] = f['label'].replace("compression", "mass effect/crowding")
            
            # RULE 4: Modifier Assignment
            if score > 0.85: f['modifier'] = "Definite"
            elif score > 0.65: f['modifier'] = "Likely"
            elif score > 0.40: f['modifier'] = "Probable"
            else: f['modifier'] = "Possible"
            
            corrected.append(f)
        return corrected
