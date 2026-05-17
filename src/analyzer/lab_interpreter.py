"""
NepalMed AI: Laboratory Result Interpreter
Analyzes lab values (LFT, CBC) and provides clinical context and simplified explanations.
"""

class LabInterpreter:
    def __init__(self):
        # Standard Reference Ranges (Nepal Clinical Standards)
        self.reference_ranges = {
            "ALT": {"min": 0, "max": 40, "unit": "U/L"},
            "AST": {"min": 0, "max": 40, "unit": "U/L"},
            "Bilirubin_Total": {"min": 0.1, "max": 1.2, "unit": "mg/dL"},
            "Platelets": {"min": 150, "max": 450, "unit": "x10^3/uL"},
            "Hemoglobin": {"min": 12, "max": 16, "unit": "g/dL"},
            "WBC": {"min": 4, "max": 11, "unit": "x10^3/uL"},
            "TSH": {"min": 0.4, "max": 4.5, "unit": "mIU/L"},
            "Free_T4": {"min": 0.8, "max": 1.8, "unit": "ng/dL"},
            "Cholesterol_Total": {"min": 0, "max": 200, "unit": "mg/dL"},
            "LDL": {"min": 0, "max": 130, "unit": "mg/dL"},
            "HDL": {"min": 40, "max": 60, "unit": "mg/dL"}
        }

    def interpret_lft(self, values):
        """
        Interprets Liver Function Test results.
        """
        results = []
        abnormal = False
        
        alt = values.get("ALT")
        if alt and alt > self.reference_ranges["ALT"]["max"]:
            results.append(f"ALT is elevated ({alt} U/L). This indicates potential liver cell injury.")
            abnormal = True
            
        bili = values.get("Bilirubin_Total")
        if bili and bili > self.reference_ranges["Bilirubin_Total"]["max"]:
            results.append(f"Total Bilirubin is high ({bili} mg/dL). This explains clinical jaundice (yellowing).")
            abnormal = True

        summary = "Normal Liver Function Test." if not abnormal else "Abnormal LFT detected."
        
        return {
            "summary": summary,
            "details": results,
            "nepali_explanation": self._get_nepali_explanation("LFT", abnormal)
        }

    def interpret_thyroid(self, values):
        """
        Interprets Thyroid Function Test (TFT) results.
        """
        results = []
        abnormal = False
        tsh = values.get("TSH")
        t4 = values.get("Free_T4")
        
        if tsh:
            if tsh > self.reference_ranges["TSH"]["max"]:
                results.append(f"TSH is high ({tsh}). Possible Hypothyroidism.")
                abnormal = True
            elif tsh < self.reference_ranges["TSH"]["min"]:
                results.append(f"TSH is low ({tsh}). Possible Hyperthyroidism.")
                abnormal = True
        
        summary = "Normal Thyroid Function." if not abnormal else "Abnormal Thyroid Profile."
        return {
            "summary": summary,
            "details": results,
            "nepali_explanation": self._get_nepali_explanation("Thyroid", abnormal)
        }

    def interpret_lipids(self, values):
        """
        Interprets Lipid Profile results.
        """
        results = []
        abnormal = False
        ldl = values.get("LDL")
        hdl = values.get("HDL")
        
        if ldl and ldl > self.reference_ranges["LDL"]["max"]:
            results.append(f"LDL (Bad Cholesterol) is high ({ldl}). Increased cardiovascular risk.")
            abnormal = True
        if hdl and hdl < self.reference_ranges["HDL"]["min"]:
            results.append(f"HDL (Good Cholesterol) is low ({hdl}).")
            abnormal = True
            
        summary = "Normal Lipid Profile." if not abnormal else "Elevated Lipid Levels."
        return {
            "summary": summary,
            "details": results,
            "nepali_explanation": self._get_nepali_explanation("Lipid", abnormal)
        }

    def interpret_cbc(self, values):
        """
        Interprets Complete Blood Count results.
        """
        results = []
        pancytopenia = False
        
        wbc = values.get("WBC")
        hb = values.get("Hemoglobin")
        plt = values.get("Platelets")
        
        # Check for Pancytopenia (Key indicator for Kala-azar in Nepal)
        if wbc and wbc < 4 and hb and hb < 10 and plt and plt < 100:
            pancytopenia = True
            results.append("WARNING: Pancytopenia detected (Low WBC, Hb, and Platelets).")

        if plt and plt < 150:
            results.append(f"Thrombocytopenia (Low Platelets: {plt}). Consider Dengue if acute fever is present.")

        summary = "Abnormal CBC pattern." if results else "Routine CBC."
        
        return {
            "summary": summary,
            "details": results,
            "nepali_explanation": self._get_nepali_explanation("CBC", bool(results), pancytopenia)
        }

    def _get_nepali_explanation(self, test_type, abnormal, pancytopenia=False):
        if not abnormal:
            return f"तपाईंको {test_type} रिपोर्ट सामान्य छ।"
        
        if test_type == "LFT":
            return "कलेजोको रिपोर्टमा केही समस्या देखिएको छ। थप परीक्षणको लागि डाक्टरसँग सल्लाह लिनुहोस्।"
        
        if test_type == "Thyroid":
            return "थाइरोइडको स्तरमा असन्तुलन देखिएको छ।"
            
        if test_type == "Lipid":
            return "रगतमा बोसो (कोलेस्ट्रोल) को मात्रा बढी देखिएको छ। खानपानमा ध्यान दिनुहोस्।"
        
        if pancytopenia:
            return "रगतमा सेतो कोशिका, रातो कोशिका र प्लेटलेट्स सबै कम देखिएका छन्। यो कालाजार जस्तो गम्भीर समस्याको लक्षण हुन सक्छ।"
            
        return "रगतको रिपोर्टमा केही कमी देखिएको छ।"

if __name__ == "__main__":
    interpreter = LabInterpreter()
    # Test LFT
    print(interpreter.interpret_lft({"ALT": 250, "Bilirubin_Total": 3.5}))
    # Test CBC
    print(interpreter.interpret_cbc({"WBC": 2.5, "Hemoglobin": 8.5, "Platelets": 80}))
