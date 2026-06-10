"""Retrieval service - finds relevant chunks for a user query."""

from config.settings import get_settings
from models.schemas import RetrievedChunk
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """Service for retrieving relevant document chunks."""

    def __init__(self) -> None:
        self._embedding_service = EmbeddingService()
        self._pinecone_service = PineconeService()
        self._top_k = get_settings().top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a given question.

        Args:
            question: User's natural language question.

        Returns:
            List of RetrievedChunk objects sorted by relevance.

        Raises:
            RuntimeError: If retrieval fails.
        """
        logger.info(f"Retrieving context for: {question[:80]}...")

        # Generate query embedding
        query_embedding = self._embedding_service.generate_embedding(question)

        # Search Pinecone
        chunks = self._pinecone_service.query_vectors(
            query_embedding=query_embedding,
            top_k=self._top_k,
        )

        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        return chunks
