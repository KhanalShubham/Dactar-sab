"""
NepalMed AI: OPD Triage Engine
Analyzes symptoms to provide clinical triage and priority levels based on MoHP Nepal STP 2078.
"""

import re

class OPDTriageEngine:
    def __init__(self):
        # Red Flag Keywords (Emergency - Immediate Referral)
        self.emergency_flags = {
            "chest_pain": [r"chest pain", r"crushing", r"heart pain"],
            "breathing": [r"shortness of breath", r"difficulty breathing", r"stridor", r"cyanosis"],
            "consciousness": [r"unconscious", r"fainting", r"syncope", r"altered mental state"],
            "trauma": [r"heavy bleeding", r"fracture", r"burn", r"poisoning", r"snake bite"],
            "neurological": [r"seizure", r"convulsion", r"sudden weakness", r"slurred speech"]
        }
        
        # Priority Keywords (Urgent - See within 24h)
        self.priority_flags = {
            "high_fever": [r"high fever", r"chills", r"rigors"],
            "abdominal": [r"severe abdominal pain", r"bloody stool", r"persistent vomiting"],
            "maternal": [r"pregnant", r"vaginal bleeding", r"decreased fetal movement", r"severe headache during pregnancy", r"swelling of hands and face"],
            "infections": [r"pancytopenia", r"splenomegaly", r"weight loss", r"persistent cough"]
        }

    def analyze_symptoms(self, symptom_text):
        """
        Analyzes symptom text and returns a triage summary.
        """
        text = symptom_text.lower()
        findings = []
        triage_level = "ROUTINE"
        reason = "Non-specific or routine symptoms."
        
        # 1. Check for Emergencies
        for category, patterns in self.emergency_flags.items():
            for p in patterns:
                if re.search(p, text):
                    findings.append(f"EMERGENCY: {category.capitalize()} detected.")
                    triage_level = "IMMEDIATE"
                    reason = f"Critical red flag detected: {p}. Immediate stabilization and referral required."
                    return self._format_result(triage_level, reason, findings)

        # 2. Check for Priority
        for category, patterns in self.priority_flags.items():
            for p in patterns:
                if re.search(p, text):
                    findings.append(f"PRIORITY: {category.capitalize()} suspected.")
                    triage_level = "URGENT"
                    reason = f"Significant symptoms detected: {p}. Prioritize clinical correlation."

        # 3. Contextual Reasoning (Nepal Specific)
        if "itch" in text and ("yellow" in text or "dark urine" in text):
            findings.append("CONTEXT: Itching with jaundice signs.")
            triage_level = "URGENT"
            reason = "Possible liver dysfunction (Hepatitis/Cholestasis). Requires LFT and ultrasound."

        if "fever" in text and ("splenomegaly" in text or "pancytopenia" in text):
            findings.append("CONTEXT: Tropical fever pattern.")
            triage_level = "URGENT"
            reason = "High suspicion for Kala-azar or Typhoid. rK39 test advised."

        return self._format_result(triage_level, reason, findings)

    def _format_result(self, level, reason, findings):
        return {
            "triage_level": level,
            "reason": reason,
            "detected_flags": findings,
            "advice": self._get_advice(level)
        }

    def _get_advice(self, level):
        advice_map = {
            "IMMEDIATE": "⚠️ REFER TO EMERGENCY DEPARTMENT IMMEDIATELY. Start stabilization protocols.",
            "URGENT": "🏥 Prioritize for medical officer review within 2-4 hours. Baseline investigations required.",
            "ROUTINE": "📅 Routine OPD review. Follow standard treatment protocols."
        }
        return advice_map.get(level, "Clinical correlation required.")

if __name__ == "__main__":
    engine = OPDTriageEngine()
    test_case = "Patient has severe chest pain and difficulty breathing."
    print(engine.analyze_symptoms(test_case))
