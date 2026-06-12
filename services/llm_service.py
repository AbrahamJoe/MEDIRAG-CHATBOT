"""LLM service - generates answers using Azure OpenAI GPT."""

from openai import AzureOpenAI

from config.settings import get_settings
from models.schemas import QueryResponse, RetrievedChunk
from services.retrieval_service import RetrievalService
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a medical knowledge assistant.

Answer ONLY using the provided context from uploaded medical documents.
Never invent or hallucinate information.
If the information is unavailable in the context, say:
"I could not find this information in the uploaded documents."

When answering:
- Be clear, concise, and accurate.
- Use bullet points for lists of symptoms, treatments, etc.
- Always cite sources at the end of your response.
- Format citations as: [Source: filename, Page X]

If asked to compare diseases, create a structured comparison."""

USER_PROMPT_TEMPLATE = """Context from medical documents:
---
{context}
---

Question: {question}

Provide a comprehensive answer based ONLY on the context above. Include citations."""


class LLMService:
    """Service for generating grounded answers with Azure OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_chat_deployment
        self._retrieval_service = RetrievalService()

    def answer_question(self, question: str) -> QueryResponse:
        """Generate an answer for a user question using RAG.

        Args:
            question: User's natural language question.

        Returns:
            QueryResponse with answer and source citations.

        Raises:
            RuntimeError: If answer generation fails.
        """
        logger.info(f"Answering question: {question[:80]}...")

        # Retrieve relevant chunks 
        chunks = self._retrieval_service.retrieve(question)

        if not chunks:
            return QueryResponse(
                answer="I could not find this information in the uploaded documents.",
                sources=[],
                query=question,
            )

        # Build context from retrieved chunks
        context = self._format_context(chunks)

        # Generate answer
        answer = self._generate_response(question, context)

        return QueryResponse(
            answer=answer,
            sources=chunks,
            query=question,
        )

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] Source: {chunk.source_file}, Page {chunk.page_number}\n"
                f"{chunk.text}\n"
            )
        return "\n".join(context_parts)

    def _generate_response(self, question: str, context: str) -> str:
        """Call Azure OpenAI to generate a response.

        Args:
            question: User's question.
            context: Formatted context from retrieved chunks.

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If API call fails.
        """
        try:
            user_message = USER_PROMPT_TEMPLATE.format(
                context=context, question=question
            )

            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            answer = response.choices[0].message.content
            logger.info("Generated answer successfully")
            return answer

        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}") from e
