"""
NepalMed AI: Document Processing Pipeline
Handles text extraction, cleaning, and clinical entity preservation for RAG.
"""

import re
import os

class MedicalDocumentProcessor:
    def __init__(self):
        # Clinical terms that should never be removed during cleaning
        self.preserved_entities = [
            r"Kala-azar", r"L-AmB", r"rK39", r"STP", r"MoHP", r"Dengue", 
            r"Pancytopenia", r"Splenomegaly", r"CES", r"Cauda Equina"
        ]

    def clean_text(self, text):
        """
        Cleans raw text while preserving medical structure.
        """
        # 1. Basic whitespace cleaning
        text = re.sub(r'\s+', ' ', text)
        
        # 2. Preserve section headers (e.g., "1. Diagnosis:")
        text = re.sub(r'(\d+\.\s+[A-Z])', r'\n\1', text)
        
        # 3. Handle common PDF artifact removals (e.g., page numbers)
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        return text.strip()

    def chunk_document(self, text, chunk_size=1000, overlap=150):
        """
        Smart clinical chunking: tries to avoid splitting in the middle of a protocol.
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                # Keep overlap from the end of the previous chunk
                current_chunk = current_chunk[-overlap:] + " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def process_folder(self, folder_path):
        """Processes all txt files in a folder and returns a list of chunks."""
        all_chunks = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    cleaned = self.clean_text(content)
                    chunks = self.chunk_document(cleaned)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({
                            "text": chunk,
                            "metadata": {"source": filename, "chunk_id": i}
                        })
        return all_chunks

if __name__ == "__main__":
    processor = MedicalDocumentProcessor()
    # test_text = "Standard Treatment Protocol 2078. Section 1. Malaria. Malaria is endemic in Terai."
    # print(processor.chunk_document(test_text))
