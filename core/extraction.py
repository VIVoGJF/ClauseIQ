from pathlib import Path
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


# Tesseract path config
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # Linux / Mac / Render / Docker
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# Helpers

def _ocr_image(img: Image.Image) -> str: # Render page to image and run Tesseract OCR on it

    return pytesseract.image_to_string(img)


def _extract_text_pymupdf(doc: fitz.Document, page: fitz.Page) -> str:
    
    extracted_text = []

    text = page.get_text("text") or ""
    if text.strip():
        extracted_text.append(text.strip())


    images = page.get_images(full=True)
    for img in images:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_text = _ocr_image(pil_image)
        if image_text.strip():
            extracted_text.append(image_text.strip())

    return "\n".join(extracted_text).strip()


def _page_to_pil(page: fitz.Page, dpi: int = 300) -> Image.Image: # Render a PyMuPDF page to a PIL Image for OCR
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat)
    img_bytes = pixmap.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))



# Core extraction
def extract_text_pages(pdf_path: str, use_ocr: bool = True) -> list[str]:
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    pages_text: list[str] = []

    with fitz.open(str(pdf_file)) as doc:
        for page_num, page in enumerate(doc, start=1):

            #PyMuPDF + OCR for images within page
            text = ""
            try:
                text = _extract_text_pymupdf(doc, page)
            except Exception as e:
                print(f"[PyMuPDF] Failed on page {page_num}: {e}")

            #OCR fallback for PyMuPDF text extraction failures
            if use_ocr and (not text.strip() or len(text.strip()) < 10):
                try:
                    print(f"[OCR] Falling back to Tesseract on page {page_num}")
                    img = _page_to_pil(page)
                    ocr_text = _ocr_image(img)
                    text = text + "\n" + ocr_text.strip()
                except Exception as e:
                    print(f"[OCR] Failed on page {page_num}: {e}")

            pages_text.append(text.strip())

    return pages_text


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> str:
   
    pages = extract_text_pages(pdf_path, use_ocr=use_ocr)

    if not pages:
        return "No readable text found."

    joined = []
    for i, p in enumerate(pages, start=1):
        joined.append(f"\n--- Page {i} ---\n{p}")
    return "\n".join(joined).strip()