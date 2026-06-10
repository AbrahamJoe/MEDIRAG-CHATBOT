# 🏥 Medical Knowledge Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** application for querying medical PDF documents. Upload medical literature and get accurate, cited answers powered by Azure OpenAI and Pinecone.

## Features

- **PDF Upload & Processing** — Upload multiple medical PDFs with automatic text extraction and chunking
- **Semantic Search** — Vector similarity search using Azure OpenAI embeddings + Pinecone
- **Grounded Answers** — GPT-generated responses strictly based on uploaded document context
- **Source Citations** — Every answer includes source file, page number, and relevance score
- **Chat History** — Persistent conversation within a session
- **Disease Comparison** — Ask comparative questions across multiple documents
- **Document Statistics** — Track PDFs, chunks, and vectors in real-time

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → Pinecone Storage
                                                            ↓
Question → Query Embedding → Pinecone Search → Top-K Chunks → GPT → Answer + Citations
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | Python 3.11 |
| Vector DB | Pinecone (Serverless) |
| Embeddings | Azure OpenAI `text-embedding-3-small` |
| LLM | Azure OpenAI GPT |
| PDF Processing | pdfplumber |
| Framework | LangChain |
| Validation | Pydantic |

## Project Structure

```
medical-rag/
├── app.py                      # Streamlit application
├── config/
│   └── settings.py             # Environment configuration
├── services/
│   ├── pdf_service.py          # PDF text extraction
│   ├── embedding_service.py    # Azure OpenAI embeddings
│   ├── pinecone_service.py     # Vector database operations
│   ├── ingestion_service.py    # End-to-end ingestion pipeline
│   ├── retrieval_service.py    # Semantic retrieval
│   └── llm_service.py          # Answer generation with GPT
├── models/
│   └── schemas.py              # Pydantic data models
├── utils/
│   └── logger.py               # Logging configuration
├── uploads/                    # Stored uploaded PDFs
├── requirements.txt
├── .env.example
└── README.md
```

## Setup & Installation

### Prerequisites

- Python 3.11+
- Azure OpenAI resource with deployments for embeddings and chat
- Pinecone account (free tier works)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd medical-rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=medical-rag
```

### 5. Azure OpenAI Setup

1. Create an Azure OpenAI resource in the Azure Portal
2. Deploy `text-embedding-3-small` model (for embeddings)
3. Deploy `gpt-4o` or `gpt-4o-mini` model (for chat)
4. Copy the endpoint and API key to your `.env` file

### 6. Pinecone Setup

1. Create a free account at [pinecone.io](https://www.pinecone.io)
2. Create an API key
3. The application will automatically create the index with:
   - **Name:** `medical-rag`
   - **Dimension:** 1536
   - **Metric:** cosine

### 7. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. **Upload PDFs** — Use the sidebar to upload medical PDF documents
2. **Process Documents** — Click "Process Documents" to ingest them into the vector database
3. **Ask Questions** — Type medical questions in the chat input
4. **View Citations** — Expand the sources section to see where answers came from

### Example Questions

- "What are the symptoms of diabetes?"
- "How can tuberculosis be prevented?"
- "Compare malaria and dengue symptoms."
- "What causes kidney disease?"
- "What treatments are available for breast cancer?"

## Screenshots

<!-- Add screenshots here -->
| Upload PDFs | Ask Questions | View Sources |
|-------------|---------------|--------------|
| *Screenshot* | *Screenshot* | *Screenshot* |

## Error Handling

The application handles:
- Empty or corrupted PDFs
- Failed uploads and processing errors
- Embedding generation failures
- Pinecone connection/query errors
- Azure OpenAI API errors
- Missing or invalid API keys

## Logging

Logs are written to `logs/app.log` with detailed information about:
- Document uploads and processing
- Embedding generation
- Pinecone operations
- Query processing
- Errors and exceptions

## License

MIT
