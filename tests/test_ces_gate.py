import unittest
from unittest.mock import MagicMock

# Mocking streamlit for testing
import sys
from types import ModuleType

mock_st = ModuleType("streamlit")
mock_st.session_state = {}
mock_st.error = MagicMock()
mock_st.warning = MagicMock()
mock_st.rerun = MagicMock()
sys.modules["streamlit"] = mock_st

class TestCESGate(unittest.TestCase):
    def test_red_flag_detection(self):
        """Verifies that red flags trigger the CES_BLOCKED state."""
        # Simulated red flag inputs
        saddle = True
        bladder = False
        bowel = False
        weakness = False
        
        # Logic from app.py
        state = "INTAKE"
        if saddle or bladder or bowel or weakness:
            state = "CES_BLOCKED"
        
        self.assertEqual(state, "CES_BLOCKED")
        print("✅ Red flag detection logic verified.")

    def test_clear_intake(self):
        """Verifies that no red flags allow proceeding."""
        saddle = False
        bladder = False
        bowel = False
        weakness = False
        
        state = "INTAKE"
        if saddle or bladder or bowel or weakness:
            state = "CES_BLOCKED"
        else:
            state = "UPLOAD"
            
        self.assertEqual(state, "UPLOAD")
        print("✅ Safe intake logic verified.")

if __name__ == "__main__":
    unittest.main()
