import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import Client

# Load the LoRA fine-tuned model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Path to the fine-tuned LoRA model

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1, vec2 = vec1.flatten(), vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

from sentence_transformers import SentenceTransformer
from chromadb import Client
import traceback

def embed_texts_with_chroma(texts, collection_name="knowledge_base"):
    """
    Get embeddings for a list of texts using the embedding model.
    Stores embeddings in a specified ChromaDB collection.
    """
    try:
        # Generate embeddings for the texts
        embeddings = embedding_model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # Setup ChromaDB client
        chroma_client = Client()

        # Create or get the collection
        if collection_name not in chroma_client.list_collections():
            chroma_client.create_collection(name=collection_name)
        collection = chroma_client.get_collection(name=collection_name)

        # Store embeddings in the ChromaDB collection
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            collection.add(
                ids=[f"text_{idx}"],
                documents=[text],
                embeddings=[embedding.tolist()]
            )
        return collection
    except Exception as e:
        print("Error details:")
        traceback.print_exc()
        raise RuntimeError(f"Failed to embed texts: {e}")

