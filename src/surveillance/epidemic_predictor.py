"""
NepalMed AI: Epidemic Predictor
Analyzes historical surveillance logs to identify rising trends and predict potential outbreaks.
"""

import json
import os
import datetime
from collections import Counter

class EpidemicPredictor:
    def __init__(self, log_path="data/surveillance/outbreak_logs.jsonl"):
        self.log_path = log_path

    def analyze_trends(self):
        """
        Analyzes recent logs to identify conditions with rising frequencies.
        """
        if not os.path.exists(self.log_path):
            return []

        recent_cases = []
        with open(self.log_path, "r") as f:
            for line in f:
                recent_cases.append(json.loads(line))

        # Filter for cases in the last 30 days (simulation)
        # In production, this would use proper datetime filtering
        
        counts = Counter([c["condition"] for c in recent_cases])
        
        predictions = []
        for condition, count in counts.items():
            if count >= 3: # Threshold for 'Outbreak Risk'
                predictions.append({
                    "condition": condition,
                    "risk_level": "HIGH" if count >= 5 else "MODERATE",
                    "reason": f"Detected {count} cases in a short period.",
                    "districts": list(set([c["district"] for c in recent_cases if c["condition"] == condition]))
                })
        
        return predictions

if __name__ == "__main__":
    predictor = EpidemicPredictor()
    print(predictor.analyze_trends())
