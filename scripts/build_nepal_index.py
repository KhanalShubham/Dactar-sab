import os
import sys
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.pdf_processor import PDFProcessor
from src.rag.build_index import IndexBuilder

def setup_logging(log_dir="data/nepal_guidelines/logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "build_index.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("NepalIndexPipeline")

def main():
    parser = argparse.ArgumentParser(description="Build Nepal Medical Guidelines RAG Index")
    parser.add_argument("--source_dir", type=str, default="data/nepal_guidelines/sources", help="Dir with PDFs")
    parser.add_argument("--output_dir", type=str, default="data/rag_index", help="Output dir for FAISS index")
    parser.add_argument("--model", type=str, default="dmis-lab/biobert-base-cased-v1.2", help="Embedding model")
    parser.add_argument("--tesseract", type=str, help="Path to tesseract.exe (Windows only)")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("Starting Nepal Medical Guidelines Indexing Pipeline")
    
    # 1. Identify PDFs
    if not os.path.exists(args.source_dir):
        logger.error(f"Source directory {args.source_dir} not found.")
        return

    pdf_files = [f for f in os.listdir(args.source_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDFs found in {args.source_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDFs to process.")

    # 2. Process PDFs
    processor = PDFProcessor(tesseract_path=args.tesseract)
    builder = IndexBuilder(model_name=args.model)
    
    from tqdm import tqdm
    all_chunks = []
    
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(args.source_dir, pdf)
        logger.info(f"Processing {pdf}...")
        
        pages_content = processor.extract_text(pdf_path)
        if not pages_content:
            logger.warning(f"No text extracted from {pdf}. Skipping.")
            continue
            
        # 3. Create Chunks
        chunks = builder.create_chunks(pages_content, source_name=pdf)
        all_chunks.extend(chunks)
        logger.info(f"Generated {len(chunks)} chunks from {pdf}")

    if not all_chunks:
        logger.error("No chunks generated from any PDF. Indexing aborted.")
        return

    # 4. Build FAISS Index
    logger.info(f"Building FAISS index for {len(all_chunks)} total chunks...")
    builder.build_faiss_index(all_chunks, output_dir=args.output_dir)
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
