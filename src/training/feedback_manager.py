"""
NepalMed AI: Feedback & Self-Training Manager
Captures doctor corrections and RLHF signals to build the national medical dataset.
"""

import json
import datetime
import os

class FeedbackManager:
    def __init__(self, buffer_path="data/training_buffer/corrections.jsonl"):
        self.buffer_path = buffer_path
        os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)

    def log_correction(self, original_input, ai_output, doctor_correction, clinical_context=None):
        """
        Logs a high-quality training pair: {Input} -> {Ground Truth Correction}.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": original_input,
            "ai_baseline": ai_output,
            "ground_truth": doctor_correction,
            "context": clinical_context or {},
            "status": "pending_review"
        }
        
        with open(self.buffer_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        print(f"✅ Correction logged to training buffer. Total: {self._get_count()}")
        return True

    def log_rlhf(self, interaction_id, helpful_score, comments=""):
        """
        Logs binary or Likert-scale feedback for Reinforcement Learning.
        """
        rlhf_path = self.buffer_path.replace("corrections.jsonl", "rlhf_signals.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "interaction_id": interaction_id,
            "score": helpful_score, # 0 or 1
            "comments": comments
        }
        
        with open(rlhf_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True

    def _get_count(self):
        if not os.path.exists(self.buffer_path):
            return 0
        with open(self.buffer_path, "r") as f:
            return sum(1 for _ in f)

if __name__ == "__main__":
    mgr = FeedbackManager()
    mgr.log_correction(
        "Patient has fever and splenomegaly.",
        "AI Suggests: Common viral fever.",
        "Doctor Correction: Suspected Kala-azar. Order rK39 test immediately per EDCD guidelines.",
        {"district": "Saptari"}
    )
