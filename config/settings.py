"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuration settings for the Medical RAG application."""

    # Azure OpenAI
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(
        default="2024-12-01-preview", env="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-small", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    azure_openai_chat_deployment: str = Field(
        default="gpt-4o", env="AZURE_OPENAI_CHAT_DEPLOYMENT"
    )

    # Pinecone
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="medical-rag", env="PINECONE_INDEX_NAME")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5

    # Pinecone Index
    pinecone_dimension: int = 1536
    pinecone_metric: str = "cosine"
    pinecone_namespace: str = "medical-docs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
