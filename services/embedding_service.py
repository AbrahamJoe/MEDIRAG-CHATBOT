"""Azure OpenAI embedding service."""

from openai import AzureOpenAI

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_embedding_deployment

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            RuntimeError: If embedding generation fails.
        """
        try:
            response = self._client.embeddings.create(
                input=text,
                model=self._deployment,
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If embedding generation fails.
        """
        try:
            # Azure OpenAI supports batch embedding up to ~2048 inputs
            # Process in smaller batches to avoid token limits
            batch_size = 100
            all_embeddings: list[list[float]] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._deployment,
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                logger.info(
                    f"Generated embeddings for batch {i // batch_size + 1} "
                    f"({len(all_embeddings)}/{len(texts)})"
                )

            return all_embeddings
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e
