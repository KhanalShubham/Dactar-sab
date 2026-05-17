"""
NepalMed AI Academy: Training Engine for Health Workers
Provides simulated clinical scenarios to train and test FCHVs and health workers.
"""

import random

class AcademyEngine:
    def __init__(self):
        self.scenarios = [
            {
                "id": "scenario_001",
                "title": "A prolonged fever in the Terai",
                "symptoms": "A 10-year-old child from Saptari has had a fever for 20 days. Their belly feels hard and large (splenomegaly).",
                "options": [
                    "Give paracetamol and send home.",
                    "Refer for rK39 test and check for Kala-azar.",
                    "Wait for another week."
                ],
                "correct_index": 1,
                "explanation": "Prolonged fever + splenomegaly in an endemic district is a classic sign of Kala-azar. The National STP 2078 requires immediate rK39 testing."
            },
            {
                "id": "scenario_002",
                "title": "Maternal Headache",
                "symptoms": "A 7-month pregnant woman has a severe headache and swelling in her hands.",
                "options": [
                    "Advise rest and more water.",
                    "Refer to the hospital immediately for BP check.",
                    "Suggest a massage."
                ],
                "correct_index": 1,
                "explanation": "Severe headache and swelling during pregnancy are 'Danger Signs' of pre-eclampsia/eclampsia. Immediate referral is mandatory."
            }
        ]

    def get_random_scenario(self):
        return random.choice(self.scenarios)

    def evaluate_answer(self, scenario_id, selected_index):
        scenario = next((s for s in self.scenarios if s["id"] == scenario_id), None)
        if not scenario:
            return False, "Scenario not found."
            
        is_correct = scenario["correct_index"] == selected_index
        return is_correct, scenario["explanation"]

if __name__ == "__main__":
    academy = AcademyEngine()
    s = academy.get_random_scenario()
    print(f"Scenario: {s['title']}\n{s['symptoms']}")
