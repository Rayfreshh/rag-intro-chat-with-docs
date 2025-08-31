# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) application that enables question-answering over a collection of news articles. The system uses ChromaDB for vector storage, OpenAI embeddings for document retrieval, and GPT-3.5-turbo for response generation.

## Architecture

**Core Components:**
- `app.py`: Main RAG application containing document processing, embedding generation, and query functionality
- `news_articles/`: Directory containing text files with AI-related news articles for indexing
- `chroma_persistent_storage/`: ChromaDB vector database storage directory
- `requirements.txt`: Python dependencies including ChromaDB, OpenAI, and supporting libraries

**RAG Pipeline:**
1. Document loading from `news_articles/` directory (app.py:44-53)
2. Text chunking with overlap for better retrieval (app.py:59-66) 
3. Embedding generation using OpenAI's text-embedding-3-small model (app.py:85-89)
4. Storage in ChromaDB with persistent client (app.py:19-24)
5. Query processing via semantic search (app.py:109-116)
6. Response generation using retrieved context (app.py:124-148)

## Development Commands

**Environment Setup:**
```bash
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
```

**Running the Application:**
```bash
python app.py  # Run the RAG script (currently processes news articles and answers hardcoded question)
```

**Environment Variables:**
- `OPENAI_API_KEY`: Required for OpenAI API access (embeddings and completions)
- Automatically sets `CHROMA_OPENAI_API_KEY` if not present

## Key Functions

- `load_documents_from_directory()`: Loads .txt files from specified directory
- `split_text()`: Chunks documents with configurable size and overlap
- `get_openai_embedding()`: Generates embeddings using OpenAI API
- `query_documents()`: Retrieves relevant chunks using semantic search
- `generate_response()`: Creates responses using retrieved context

## Current State

The application currently runs as a script that:
1. Processes news articles in the `news_articles/` directory
2. Stores embeddings in ChromaDB
3. Answers a hardcoded question about "databricks"

The codebase uses Jupyter-style cell markers (`#%%`) suggesting interactive development.