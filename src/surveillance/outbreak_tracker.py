"""
NepalMed AI: National Outbreak Tracker (Surveillance)
Logs clinical findings geographically to monitor for potential disease outbreaks.
"""

import json
import datetime
import os

class OutbreakTracker:
    def __init__(self, log_path="data/surveillance/outbreak_logs.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_case(self, district, suspected_condition, triage_level):
        """
        Logs a case for surveillance.
        Only logs URGENT or IMMEDIATE cases for outbreak monitoring.
        """
        if triage_level not in ["URGENT", "IMMEDIATE"]:
            return False
            
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "district": district,
            "condition": suspected_condition,
            "triage_level": triage_level
        }
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        print(f"📊 Surveillance Alert: {suspected_condition} logged in {district}.")
        return True

    def get_district_stats(self):
        """Returns case counts per district for the heatmap."""
        if not os.path.exists(self.log_path):
            return {}
            
        stats = {}
        with open(self.log_path, "r") as f:
            for line in f:
                case = json.loads(line)
                dist = case["district"]
                stats[dist] = stats.get(dist, 0) + 1
        return stats

if __name__ == "__main__":
    tracker = OutbreakTracker()
    tracker.log_case("Saptari", "Kala-azar", "URGENT")
    print(tracker.get_district_stats())
