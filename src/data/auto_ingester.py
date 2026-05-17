"""
NepalMed AI: Automated Knowledge Ingester
Monitors the data/raw folder and processes new medical documents into the Vector Store.
"""

import os
import time
from src.data.document_processor import MedicalDocumentProcessor
from nepalmed_ai.scripts.rag_prototype import NepalMedRAG

class AutoIngester:
    def __init__(self, raw_dir="data/raw", processed_log="data/processed_files.log"):
        self.raw_dir = raw_dir
        self.processed_log = processed_log
        self.processor = MedicalDocumentProcessor()
        self.rag = NepalMedRAG()
        
        # Ensure log exists
        if not os.path.exists(self.processed_log):
            open(self.processed_log, "a").close()

    def get_processed_files(self):
        with open(self.processed_log, "r") as f:
            return set(f.read().splitlines())

    def mark_as_processed(self, filename):
        with open(self.processed_log, "a") as f:
            f.write(filename + "\n")

    def run_sync(self):
        """
        Scans the raw directory and processes new files.
        """
        print("🔄 Scanning for new medical knowledge...")
        processed = self.get_processed_files()
        
        count = 0
        for filename in os.listdir(self.raw_dir):
            if filename.endswith(".txt") and filename not in processed:
                print(f"📖 New file detected: {filename}")
                file_path = os.path.join(self.raw_dir, filename)
                
                # Use RAG prototype to ingest (adapted for the processor)
                # In a real scenario, this would call a more robust ingestion API
                self.rag.ingest_document(file_path) # Assumes ingest_document handles txt now
                
                self.mark_as_processed(filename)
                count += 1
        
        if count > 0:
            print(f"✅ Auto-ingestion complete. Added {count} documents.")
        else:
            print("💤 No new documents found.")

if __name__ == "__main__":
    ingester = AutoIngester()
    ingester.run_sync()
