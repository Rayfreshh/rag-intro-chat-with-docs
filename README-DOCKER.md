# RAG Microservice Docker Deployment

This directory contains a containerized RAG (Retrieval-Augmented Generation) microservice built with FastAPI.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key

### Environment Setup

1. Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### Running with Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the service
docker-compose down
```

### Running with Docker directly

```bash
# Build the image
docker build -t rag-microservice .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key_here \
  -v $(pwd)/chroma_persistent_storage:/app/chroma_persistent_storage \
  -v $(pwd)/news_articles:/app/news_articles \
  --name rag-service \
  rag-microservice
```

## API Endpoints

### Health Check
- **GET** `/` - Returns HTML interface

### Question Answering
- **POST** `/ask`
  - Body: `{"question": "your question here"}`
  - Response: `{"answer": "AI generated response"}`

### Example Usage

```bash
# Test the API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'
```

## Architecture

- **FastAPI** web framework for REST API
- **ChromaDB** for vector storage and similarity search
- **OpenAI** embeddings (text-embedding-3-small) and completions (GPT-3.5-turbo)
- **Persistent storage** for vector database
- **PDF support** via PyPDF2

## Production Considerations

- The container includes health checks
- Persistent volumes for database storage
- Environment-based configuration
- Optimized Docker layers for faster rebuilds