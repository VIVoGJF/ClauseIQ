from pathlib import Path
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# -------------------------
# Tesseract path config
# -------------------------
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # Linux / Mac / Render / Docker
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# -------------------------
# Internal helpers
# -------------------------
def _extract_text_pymupdf(page: fitz.Page) -> str:
    """Extract selectable text from a PyMuPDF page object."""
    return page.get_text("text") or ""


def _page_to_pil(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image for OCR."""
    zoom = dpi / 72  # 72 is PyMuPDF's default DPI
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat)
    img_bytes = pixmap.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def _ocr_page(page: fitz.Page) -> str:
    """Render page to image and run Tesseract OCR on it."""
    img = _page_to_pil(page)
    return pytesseract.image_to_string(img)


# -------------------------
# Core extraction
# -------------------------
def extract_text_pages(pdf_path: str, use_ocr: bool = True) -> list[str]:
    """
    Extract text per page from a PDF.

    Strategy per page:
        1. Try PyMuPDF text extraction (fast, layout-aware).
        2. If PyMuPDF fails or returns < 10 chars and use_ocr=True,
           fall back to Tesseract OCR via PyMuPDF page rendering.
        3. If OCR also fails, return empty string for that page (no crash).

    Args:
        pdf_path: Path to the PDF file.
        use_ocr:  Whether to fall back to Tesseract OCR. Default True.

    Returns:
        list[str] — one string per page, stripped.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    pages_text: list[str] = []

    with fitz.open(str(pdf_file)) as doc:
        for page_num, page in enumerate(doc, start=1):

            # ── Step 1: PyMuPDF ──────────────────────────────────────────
            text = ""
            try:
                text = _extract_text_pymupdf(page)
            except Exception as e:
                print(f"[PyMuPDF] Failed on page {page_num}: {e}")

            # ── Step 2: OCR fallback ──────────────────────────────────────
            if use_ocr and (not text.strip() or len(text.strip()) < 10):
                try:
                    print(f"[OCR] Falling back to Tesseract on page {page_num}")
                    ocr_text = _ocr_page(page)
                    text = text + "\n" + ocr_text.strip()
                except Exception as e:
                    print(f"[OCR] Failed on page {page_num}: {e}")

            pages_text.append(text.strip())

    return pages_text


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> str:
    """
    Convenience wrapper — returns full text joined with page markers.

    Returns:
        Single string with --- Page N --- markers between pages.
    """
    pages = extract_text_pages(pdf_path, use_ocr=use_ocr)

    if not pages:
        return "No readable text found."

    joined = []
    for i, p in enumerate(pages, start=1):
        joined.append(f"\n--- Page {i} ---\n{p}")
    return "\n".join(joined).strip()