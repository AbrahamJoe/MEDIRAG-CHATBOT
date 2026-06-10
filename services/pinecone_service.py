"""Pinecone vector database service."""

from pinecone import Pinecone, ServerlessSpec

from config.settings import get_settings
from models.schemas import RetrievedChunk, VectorRecord
from utils.logger import get_logger

logger = get_logger(__name__)


class PineconeService:
    """Service for managing vectors in Pinecone."""

    def __init__(self) -> None:
        settings = get_settings()
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._namespace = settings.pinecone_namespace
        self._dimension = settings.pinecone_dimension
        self._metric = settings.pinecone_metric
        self._index = None

    def _get_index(self):
        """Get or create the Pinecone index."""
        if self._index is None:
            self.create_index_if_not_exists()
            self._index = self._pc.Index(self._index_name)
        return self._index

    def create_index_if_not_exists(self) -> None:
        """Create the Pinecone index if it does not already exist, or recreate if dimension mismatches."""
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]

        if self._index_name in existing_indexes:
            # Check dimension matches
            desc = self._pc.describe_index(self._index_name)
            if desc.dimension != self._dimension:
                logger.warning(
                    f"Index '{self._index_name}' has dimension {desc.dimension}, "
                    f"expected {self._dimension}. Deleting and recreating."
                )
                self._pc.delete_index(self._index_name)
            else:
                logger.info(f"Index '{self._index_name}' already exists.")
                return

        logger.info(f"Creating Pinecone index: {self._index_name}")
        self._pc.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"Index '{self._index_name}' created successfully.")

    def upsert_vectors(self, vectors: list[VectorRecord]) -> int:
        """Upsert vectors into Pinecone.

        Args:
            vectors: List of VectorRecord objects to upsert.

        Returns:
            Number of vectors upserted.

        Raises:
            RuntimeError: If upsert fails.
        """
        try:
            index = self._get_index()
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                records = [
                    {
                        "id": v.id,
                        "values": v.values,
                        "metadata": v.metadata,
                    }
                    for v in batch
                ]
                index.upsert(vectors=records, namespace=self._namespace)
                total_upserted += len(batch)
                logger.info(
                    f"Upserted batch: {total_upserted}/{len(vectors)} vectors"
                )

            return total_upserted
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            raise RuntimeError(f"Failed to upsert vectors: {e}") from e

    def query_vectors(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Query Pinecone for similar vectors.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of RetrievedChunk objects.

        Raises:
            RuntimeError: If query fails.
        """
        try:
            index = self._get_index()
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self._namespace,
            )

            chunks: list[RetrievedChunk] = []
            for match in results.matches:
                metadata = match.metadata or {}
                chunks.append(
                    RetrievedChunk(
                        text=metadata.get("chunk_text", ""),
                        source_file=metadata.get("source_file", "Unknown"),
                        page_number=metadata.get("page_number", 0),
                        score=match.score,
                    )
                )

            logger.info(f"Retrieved {len(chunks)} chunks from Pinecone")
            return chunks
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            raise RuntimeError(f"Failed to query vectors: {e}") from e

    def get_index_stats(self) -> dict:
        """Get index statistics.

        Returns:
            Dictionary with index stats including total vector count.
        """
        try:
            index = self._get_index()
            stats = index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespaces": {
                    ns: info.vector_count
                    for ns, info in (stats.namespaces or {}).items()
                },
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"total_vectors": 0, "namespaces": {}}
