"""PDF document processing service."""

import io
from pathlib import Path

import pdfplumber

from models.schemas import PageContent
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFService:
    """Service for extracting text from PDF documents."""

    def extract_text(self, file_bytes: bytes, filename: str) -> list[PageContent]:
        """Extract text from a PDF file page by page.

        Args:
            file_bytes: Raw bytes of the PDF file.
            filename: Original filename for logging.

        Returns:
            List of PageContent objects with page number and text.

        Raises:
            ValueError: If the PDF is empty or contains no extractable text.
        """
        logger.info(f"Extracting text from: {filename}")
        pages: list[PageContent] = []

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                if not pdf.pages:
                    raise ValueError(f"PDF '{filename}' contains no pages.")

                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:
                        pages.append(PageContent(page_number=i, text=text))

            if not pages:
                raise ValueError(
                    f"PDF '{filename}' contains no extractable text."
                )

            logger.info(
                f"Extracted {len(pages)} pages with text from '{filename}'"
            )
            return pages

        except Exception as e:
            logger.error(f"Failed to extract text from '{filename}': {e}")
            raise

    def save_upload(self, file_bytes: bytes, filename: str) -> Path:
        """Save uploaded file to the uploads directory.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original filename.

        Returns:
            Path to the saved file.
        """
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / filename
        file_path.write_bytes(file_bytes)
        logger.info(f"Saved upload: {file_path}")
        return file_path
