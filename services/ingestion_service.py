"""Document ingestion service - orchestrates the full PDF-to-vector pipeline."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from models.schemas import DocumentChunk, PageContent, VectorRecord
from services.embedding_service import EmbeddingService
from services.pdf_service import PDFService
from services.pinecone_service import PineconeService
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """Orchestrates PDF ingestion: extract, chunk, embed, store."""

    def __init__(self) -> None:
        self._pdf_service = PDFService()
        self._embedding_service = EmbeddingService()
        self._pinecone_service = PineconeService()
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> int:
        """Ingest a PDF document end-to-end.

        Args:
            file_bytes: Raw PDF file bytes.
            filename: Original filename.

        Returns:
            Number of vectors stored.

        Raises:
            ValueError: If PDF is empty or invalid.
            RuntimeError: If any pipeline step fails.
        """
        # 1. Save upload
        self._pdf_service.save_upload(file_bytes, filename)

        # 2. Extract text
        pages = self._pdf_service.extract_text(file_bytes, filename)
        logger.info(f"Extracted {len(pages)} pages from '{filename}'")

        # 3. Chunk text
        chunks = self._chunk_pages(pages, filename)
        logger.info(f"Created {len(chunks)} chunks from '{filename}'")

        # 4. Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_service.generate_embeddings_batch(texts)
        logger.info(f"Generated {len(embeddings)} embeddings for '{filename}'")

        # 5. Prepare vector records
        vectors = [
            VectorRecord(
                id=chunk.chunk_id,
                values=embedding,
                metadata={
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_text": chunk.text,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # 6. Upsert to Pinecone
        count = self._pinecone_service.upsert_vectors(vectors)
        logger.info(f"Stored {count} vectors for '{filename}'")
        return count

    def _chunk_pages(
        self, pages: list[PageContent], filename: str
    ) -> list[DocumentChunk]:
        """Split pages into chunks with metadata.

        Args:
            pages: List of extracted page content.
            filename: Source filename for metadata.

        Returns:
            List of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        # Clean filename for use in IDs
        clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename.rsplit(".", 1)[0])

        for page in pages:
            page_chunks = self._splitter.split_text(page.text)
            for idx, chunk_text in enumerate(page_chunks):
                chunk_id = f"{clean_name}_page{page.page_number}_chunk{idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_file=filename,
                        page_number=page.page_number,
                    )
                )

        return chunks
