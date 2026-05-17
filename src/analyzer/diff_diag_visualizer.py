"""
NepalMed AI: Differential Diagnosis Visualizer
Calculates and visualizes potential diagnoses based on symptom patterns and Nepal-specific prevalence.
"""

class DiffDiagVisualizer:
    def __init__(self):
        # Weighted symptom mapping for common conditions in Nepal
        self.knowledge_base = {
            "Kala-azar": {
                "symptoms": ["prolonged fever", "splenomegaly", "weight loss", "pancytopenia"],
                "prevalence": "Endemic in Terai districts (Saptari, Siraha, etc.)",
                "base_weight": 0.8
            },
            "Dengue": {
                "symptoms": ["acute fever", "severe headache", "joint pain", "thrombocytopenia", "rash"],
                "prevalence": "Seasonal outbreaks (Post-monsoon)",
                "base_weight": 0.7
            },
            "Typhoid": {
                "symptoms": ["step-ladder fever", "abdominal pain", "constipation", "headache", "coated tongue"],
                "prevalence": "Common in urban areas with water sanitation issues",
                "base_weight": 0.6
            },
            "Hepatitis": {
                "symptoms": ["jaundice", "dark urine", "itching", "nausea", "elevated ALT"],
                "prevalence": "Variable; often food/water-borne (Hep A/E)",
                "base_weight": 0.6
            }
        }

    def calculate_differentials(self, symptoms_text):
        """
        Calculates likelihood scores for different diagnoses.
        """
        text = symptoms_text.lower()
        results = []
        
        for condition, data in self.knowledge_base.items():
            matches = 0
            found_symptoms = []
            for s in data["symptoms"]:
                if s in text:
                    matches += 1
                    found_symptoms.append(s)
            
            if matches > 0:
                # Basic scoring: Match ratio * base weight
                score = (matches / len(data["symptoms"])) * data["base_weight"]
                
                # Confidence Label
                label = "LOW"
                if score > 0.5: label = "MEDIUM"
                if score > 0.75: label = "HIGH"
                
                results.append({
                    "condition": condition,
                    "likelihood": round(score * 100, 1),
                    "confidence": label,
                    "matched_symptoms": found_symptoms,
                    "context": data["prevalence"]
                })
        
        # Sort by likelihood
        results.sort(key=lambda x: x["likelihood"], reverse=True)
        return results

if __name__ == "__main__":
    viz = DiffDiagVisualizer()
    print(viz.calculate_differentials("Patient has prolonged fever and weight loss and splenomegaly"))
