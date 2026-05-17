"""
NepalMed AI: Synthetic Clinical Case Generator
Generates Q&A pairs and case studies from medical guidelines for model fine-tuning.
"""

import json
import random

class SyntheticCaseGenerator:
    def __init__(self, templates_path=None):
        # Basic patterns to generate variety
        self.patient_templates = [
            "A {age} year old {gender} from {district} presents with {symptoms}.",
            "A patient ({age}y, {gender}) from {district} has {symptoms}.",
            "Emergency referral: {age}y {gender} with {symptoms} from {district}."
        ]
        
        self.districts = ["Saptari", "Siraha", "Dhanusha", "Humla", "Mustang", "Kathmandu"]

    def generate_kala_azar_cases(self, count=10):
        """
        Generates synthetic Kala-azar cases based on EDCD protocols.
        """
        cases = []
        for _ in range(count):
            age = random.randint(5, 75)
            gender = random.choice(["male", "female"])
            district = random.choice(["Saptari", "Siraha", "Dhanusha"]) # Endemic areas
            
            case_input = random.choice(self.patient_templates).format(
                age=age, gender=gender, district=district,
                symptoms="fever for 15 days, splenomegaly, and pancytopenia"
            )
            
            case_output = (
                f"Based on the patient's residence in {district} and clinical presentation of prolonged fever and pancytopenia, "
                "Visceral Leishmaniasis (Kala-azar) is highly suspected. "
                "Plan: Perform rK39 rapid diagnostic test. If positive, initiate single-dose Liposomal Amphotericin B (L-AmB) "
                "as per National Elimination Guidelines."
            )
            
            cases.append({
                "instruction": "Provide a clinical assessment and plan.",
                "input": case_input,
                "output": case_output,
                "metadata": {"specialty": "Infectious Disease", "type": "Synthetic"}
            })
            
        return cases

    def save_dataset(self, cases, filename="data/training_buffer/synthetic_cases.jsonl"):
        """Saves the generated cases for training."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case) + "\n")
        print(f"✅ Generated {len(cases)} synthetic cases.")

if __name__ == "__main__":
    import os
    gen = SyntheticCaseGenerator()
    data = gen.generate_kala_azar_cases(5)
    gen.save_dataset(data)
