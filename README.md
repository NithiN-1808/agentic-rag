# Agentic RAG Chat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)

A fully custom chatbot built with Agentic RAG (Retrieval-Augmented Generation), combining OpenAI models with a local knowledge base for accurate, context-aware, and explainable responses. Features a lightweight, dependency-free frontend and a streamlined FastAPI backend for complete control and simplicity.

![Demo](demo.gif)


## Features

- Pure HTML/CSS/JavaScript frontend with no external dependencies
- FastAPI backend with OpenAI integration
- Agentic RAG implementation with:
  - Context retrieval using embeddings and cosine similarity
  - Step-by-step reasoning with Chain of Thought
  - Function calling for dynamic context retrieval
- Comprehensive error handling and logging
- Type-safe implementation with Python type hints
- Configurable through environment variables

## Project Structure

```
agentic_rag/
├── backend/
│   ├── embeddings.py    # Embedding and similarity functions
│   ├── rag_engine.py
    |__ fine_tune_lora.py    # Core RAG implementation
│   └── server.py        # FastAPI server
├── frontend/
│   └── index.html       # Web interface
├── requirements.txt     # Python dependencies
├── .env.sample         # Sample environment variables
└── README.md           # Documentation
```

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Git (for version control)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NithiN-1808/agentic-rag.git
cd agentic_rag
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.sample .env
```
Then edit `.env` with your configuration:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini  # or another compatible model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small # or another compatible model
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python server.py
```

2. Access the frontend:
- Option 1: Open `frontend/index.html` directly in your web browser
- Option 2: Serve using Python's built-in server:
```bash
cd frontend
python -m http.server 3000
```

Then visit http://localhost:3000 in your browser.



## Configuration

The application can be configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | Your OpenAI API key | Required |
| OPENAI_MODEL | OpenAI model to use | gpt-4o-mini |
| HOST | Backend server host | 0.0.0.0 |
| PORT | Backend server port | 8000 |



## Security

- Never commit your `.env` file or API keys
- Keep dependencies updated
- Follow security best practices for production deployment
- Report security issues through GitHub's security advisory



## Acknowledgments

- OpenAI for their API and models
- FastAPI framework
- Contributors and maintainers
