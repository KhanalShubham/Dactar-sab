"""
NepalMed AI: Patient Longitudinal Record
Stores and retrieves historical clinical visits for patients to track health trends over time.
"""

import json
import os
import datetime

class PatientHistory:
    def __init__(self, data_dir="data/patients"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_path(self, patient_id):
        return os.path.join(self.data_dir, f"{patient_id}.jsonl")

    def log_visit(self, patient_id, clinical_data):
        """
        Logs a new visit for a patient.
        """
        path = self._get_path(patient_id)
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data": clinical_data
        }
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True

    def get_history(self, patient_id):
        """
        Retrieves all past visits for a patient.
        """
        path = self._get_path(patient_id)
        if not os.path.exists(path):
            return []
            
        history = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                history.append(json.loads(line))
        return history

if __name__ == "__main__":
    ph = PatientHistory()
    p_id = "P-777"
    ph.log_visit(p_id, {"symptoms": "Fever", "triage": "URGENT"})
    print(ph.get_history(p_id))
