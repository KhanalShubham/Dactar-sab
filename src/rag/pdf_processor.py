import os
import logging
import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_path
try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger("PDFProcessor")

class PDFProcessor:
    """Handles text extraction from digital and scanned PDFs."""
    
    def __init__(self, tesseract_path=None):
        if tesseract_path and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, pdf_path):
        """
        Extracts text page-by-pate. Tries digital extraction first,
        falls back to OCR if digital text is sparse.
        """
        pages_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    # Heuristic: If less than 100 chars, try OCR fallback for this page
                    if not text or len(text.strip()) < 100:
                        logger.info(f"Page {i+1} of {pdf_path} looks scanned. Attempting OCR...")
                        text = self._ocr_page(pdf_path, i)
                    
                    if text:
                        pages_content.append({
                            "page_no": i + 1,
                            "text": self._clean_text(text)
                        })
            
            logger.info(f"Successfully extracted {len(pages_content)} pages from {pdf_path}")
            return pages_content
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return []

    def _ocr_page(self, pdf_path, page_index):
        """Converts a specific PDF page to image and runs OCR."""
        if not pytesseract:
            logger.warning("pytesseract not installed. Skipping OCR fallback.")
            return ""
            
        try:
            # pdf2image page indices are 1-based or we can specify first_page/last_page
            images = convert_from_path(pdf_path, first_page=page_index+1, last_page=page_index+1, dpi=300)
            if not images:
                return ""
            
            # Run OCR
            text = pytesseract.image_to_string(images[0])
            return text
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path} page {page_index+1}: {e}")
            return ""

    def _clean_text(self, text):
        """Normalises whitespace and basic cleanup."""
        if not text:
            return ""
        # Replace multiple spaces with single space
        text = " ".join(text.split())
        # Basic OCR error cleanup (can be expanded)
        # text = text.replace("rn", "m") 
        return text.strip()
