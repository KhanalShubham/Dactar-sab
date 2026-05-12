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
