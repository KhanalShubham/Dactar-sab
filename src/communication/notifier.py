"""
NepalMed AI: Clinical Notification System
Simulates sending SMS/Email alerts to referral hospitals for high-priority cases.
"""

import datetime

class ClinicalNotifier:
    def __init__(self):
        # Simulated database of referral hospitals and their contact points
        self.referral_registry = {
            "Saptari": {"hospital": "Gajendra Narayan Singh Hospital", "contact": "+977-980XXXXXXX"},
            "Kathmandu": {"hospital": "T.U. Teaching Hospital (TUTH)", "contact": "+977-01-441XXXX"},
            "Morang": {"hospital": "Koshi Hospital", "contact": "+977-021-XXXXXX"}
        }

    def notify_referral(self, district, patient_name, condition, priority):
        """
        Simulates sending a notification to the local referral center.
        """
        registry = self.referral_registry.get(district, {"hospital": "District Health Office", "contact": "Regional Admin"})
        
        timestamp = datetime.datetime.now().strftime("%H:%M")
        message = f"""
🚨 NEPALMED AI ALERT 🚨
Time: {timestamp}
Priority: {priority}
Patient: {patient_name}
Suspected: {condition}
Action: Patient referred to {registry['hospital']}.
Please prepare for intake.
        """
        
        print(f"📡 SENDING SMS to {registry['contact']}...")
        print(message)
        return True, message

if __name__ == "__main__":
    notifier = ClinicalNotifier()
    notifier.notify_referral("Saptari", "Patient-402", "Kala-azar", "URGENT")
