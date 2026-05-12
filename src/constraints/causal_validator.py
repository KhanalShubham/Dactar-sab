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
