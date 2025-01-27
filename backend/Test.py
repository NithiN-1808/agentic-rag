from sentence_transformers import SentenceTransformer

def test_model():
    try:
        # Load a pre-trained SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode a sample sentence
        embeddings = model.encode(["Test sentence"])
        
        print("Embeddings generated successfully!")
        print(embeddings)
    except Exception as e:
        print(f"Error encountered: {e}")

if __name__ == "__main__":
    test_model()
