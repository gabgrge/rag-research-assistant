# Research Assistant (RAG)

A local Research Assistant (RAG) with a Streamlit UI, Chroma vector store, and a simple update pipeline for converting and indexing documents from a synced Google Drive folder.

## Features

- Local Streamlit-based chat interface for conversational document search
- Retrieval-Augmented Generation (RAG) workflow over locally indexed documents
- Chroma vector database with hierarchical chunking for semantic retrieval
- Automated document ingestion pipeline for PDFs, Office files, and legacy formats
- Source-backed answers with citations and traceability
- Admin utilities for index backup, reset, and local maintenance

## Quickstart

1. Copy `.env.example` to `.env` and fill in the values.
2. Build the Docker environment: `make docker-install`
3. Start the app: `make docker-up`

## Documentation

[https://gabgrge.github.io/rag-research-assistant/](https://gabgrge.github.io/rag-research-assistant/)

## License

Copyright © 2026 gabgrge. All Rights Reserved.