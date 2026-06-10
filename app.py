"""Medical Knowledge Assistant - Streamlit Application."""

import streamlit as st
from datetime import datetime

from services.ingestion_service import IngestionService
from services.llm_service import LLMService
from services.pinecone_service import PineconeService
from utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="MEDIRAG-CHATBOT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #566573;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F8F9F9;
        border-left: 4px solid #2E86C1;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .chat-user {
        background-color: #EBF5FB;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background-color: #F4F6F6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "total_vectors" not in st.session_state:
        st.session_state.total_vectors = 0
        # Load existing stats from Pinecone on first load
        try:
            pinecone_service = PineconeService()
            stats = pinecone_service.get_index_stats()
            st.session_state.total_vectors = stats.get("total_vectors", 0)
        except Exception:
            pass
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0


def render_sidebar():
    """Render the sidebar with upload and stats."""
    with st.sidebar:
        st.markdown("## 🏥 MEDIRAG-CHATBOT")
        st.markdown("---")

        # PDF Upload Section
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload medical PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF files containing medical information.",
        )

        if uploaded_files:
            if st.button("📤 Process Documents", use_container_width=True):
                process_uploads(uploaded_files)

        st.markdown("---")

        # Document Statistics
        st.markdown("### 📊 Document Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PDFs", len(st.session_state.uploaded_files))
        with col2:
            st.metric("Vectors", st.session_state.total_vectors)

        st.metric("Total Chunks", st.session_state.total_chunks)

        # Uploaded files list
        if st.session_state.uploaded_files:
            st.markdown("### 📁 Uploaded Files")
            for fname in st.session_state.uploaded_files:
                st.markdown(f"• {fname}")

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Refresh stats
        if st.button("🔄 Refresh Stats", use_container_width=True):
            refresh_stats()


def process_uploads(uploaded_files):
    """Process uploaded PDF files through the ingestion pipeline."""
    ingestion_service = IngestionService()

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name

        if filename in st.session_state.uploaded_files:
            status_text.warning(f"'{filename}' already uploaded. Skipping.")
            continue

        status_text.info(f"Processing: {filename}...")

        try:
            file_bytes = uploaded_file.read()
            vectors_stored = ingestion_service.ingest_pdf(file_bytes, filename)

            st.session_state.uploaded_files.append(filename)
            st.session_state.total_vectors += vectors_stored
            st.session_state.total_chunks += vectors_stored

            status_text.success(f"✅ {filename}: {vectors_stored} vectors stored")
            logger.info(f"Successfully ingested '{filename}'")

        except ValueError as e:
            status_text.error(f"❌ {filename}: {str(e)}")
            logger.error(f"Validation error for '{filename}': {e}")
        except RuntimeError as e:
            status_text.error(f"❌ {filename}: Processing failed")
            logger.error(f"Runtime error for '{filename}': {e}")
        except Exception as e:
            status_text.error(f"❌ {filename}: Unexpected error")
            logger.error(f"Unexpected error for '{filename}': {e}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()
    st.rerun()


def refresh_stats():
    """Refresh statistics from Pinecone."""
    try:
        pinecone_service = PineconeService()
        stats = pinecone_service.get_index_stats()
        st.session_state.total_vectors = stats.get("total_vectors", 0)
        st.rerun()
    except Exception as e:
        st.sidebar.error("Failed to refresh stats.")
        logger.error(f"Stats refresh failed: {e}")


def render_chat():
    """Render the main chat interface."""
    st.markdown('<p class="main-header">Medical Knowledge Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about uploaded medical documents. '
        "Get accurate, cited answers powered by RAG.</p>",
        unsafe_allow_html=True,
    )

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(msg["content"])

                # Show sources
                if msg.get("sources"):
                    with st.expander("📚 View Sources & Citations"):
                        for source in msg["sources"]:
                            st.markdown(
                                f'<div class="source-box">'
                                f"<strong>{source['source_file']}</strong> — Page {source['page_number']} "
                                f"(Score: {source['score']:.3f})<br>"
                                f"<em>{source['text'][:300]}...</em>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

    # Question input
    question = st.chat_input("Ask a medical question about your uploaded documents...")

    if question:
        handle_question(question)


def handle_question(question: str):
    """Process user question and generate response."""
    # Add user message to history
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "timestamp": datetime.now().isoformat()}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Check if documents have been uploaded (session or Pinecone)
    if not st.session_state.uploaded_files and st.session_state.total_vectors == 0:
        # Check Pinecone for existing vectors
        try:
            pinecone_service = PineconeService()
            stats = pinecone_service.get_index_stats()
            st.session_state.total_vectors = stats.get("total_vectors", 0)
        except Exception:
            pass

    if not st.session_state.uploaded_files and st.session_state.total_vectors == 0:
        no_docs_msg = "Please upload medical PDF documents first before asking questions."
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": no_docs_msg,
                "sources": [],
                "timestamp": datetime.now().isoformat(),
            }
        )
        with st.chat_message("assistant", avatar="🏥"):
            st.warning(no_docs_msg)
        return

    # Generate response
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                llm_service = LLMService()
                response = llm_service.answer_question(question)

                st.markdown(response.answer)

                # Format sources for storage
                sources_data = [
                    {
                        "source_file": s.source_file,
                        "page_number": s.page_number,
                        "score": s.score,
                        "text": s.text,
                    }
                    for s in response.sources
                ]

                # Show sources
                if response.sources:
                    with st.expander("📚 View Sources & Citations"):
                        for source in response.sources:
                            st.markdown(
                                f'<div class="source-box">'
                                f"<strong>{source.source_file}</strong> — Page {source.page_number} "
                                f"(Score: {source.score:.3f})<br>"
                                f"<em>{source.text[:300]}...</em>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # Store in history
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                error_msg = "An error occurred while generating the answer. Please try again."
                st.error(error_msg)
                logger.error(f"Question handling failed: {e}")
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "timestamp": datetime.now().isoformat(),
                    }
                )


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
