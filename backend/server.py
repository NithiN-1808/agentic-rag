from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Agentic RAG Chat API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
try:
    rag_engine = RAGEngine()
    logger.info("RAG Engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {e}")
    raise RuntimeError("Server initialization failed due to RAG Engine error.")

# Define request models
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint to process user messages and return reasoned responses.
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
        response = rag_engine.get_response(request.message)
        return {
            "message": "Query processed successfully.",
            "response": response
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")
