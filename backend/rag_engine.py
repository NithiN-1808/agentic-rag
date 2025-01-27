import logging
import numpy as np
import openai
import requests
import wikipediaapi
from chromadb import Client
from llama_index.core import SimpleKeywordTableIndex, Document
from embeddings import embed_texts_with_chroma
from rouge import Rouge

# Initialize logging and OpenAI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_COLLECTION_NAME = "knowledge_base"
OPENAI_API_KEY = "sk-your-api-key-here"  # Set your OpenAI API Key
openai.api_key = OPENAI_API_KEY  # Set the API key for OpenAI

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="MyRAGEngineBot/1.0 (https://mywebsite.com/; myemail@example.com)"
)


# Function to fetch articles from PubMed (E-utilities)
def fetch_pubmed_articles(query, max_results=5):
    """Fetches articles from PubMed based on a search query."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'usehistory': 'y',
        'retmode': 'xml'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text  # This returns the XML response
    else:
        logger.error("Failed to fetch PubMed articles.")
        return []

# Function to get summaries from Wikipedia
def get_wikipedia_summary(query):
    """Fetches the summary of a Wikipedia page."""
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    else:
        return "No relevant Wikipedia page found."

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine with external knowledge sources."""
        self.chroma_client = Client()
        
        # External knowledge sources
        self.pubmed_articles = fetch_pubmed_articles("cancer", 5)  # Example query for PubMed
        self.wikipedia_articles = [
            get_wikipedia_summary("Machine learning"),
            get_wikipedia_summary("Artificial intelligence"),
            get_wikipedia_summary("Healthcare"),
        ]
        
        # Combine external data with static data
        self.data = [
            *self.pubmed_articles,  # Adding PubMed articles to the knowledge base
            *self.wikipedia_articles,  # Adding Wikipedia summaries to the knowledge base
            "Python is a versatile programming language.",
            "GPT-4 supports function calling for seamless tool integration.",
            "Transformers are widely used for NLP tasks.",
            "Kubernetes is essential for container orchestration.",
            "The heart is a vital organ responsible for pumping blood.",
            "COVID-19 is caused by the SARS-CoV-2 virus.",
            "Insulin resistance is a hallmark of Type 2 diabetes.",
            "The liver detoxifies harmful substances in the body.",
            "Lung cancer is one of the leading causes of cancer deaths.",
            "Alzheimer's disease is a progressive neurodegenerative disorder.",
            "Artificial intelligence is revolutionizing various industries.",
            "Quantum computing leverages the principles of quantum mechanics.",
            "Deep learning models can be used for image recognition.",
            "ChromaDB helps store and retrieve document embeddings efficiently.",
            "LoRA techniques improve model training by adding low-rank layers.",
            "LangChain enables seamless integration of NLP tasks into pipelines.",
            "Chronic stress can lead to various health problems, including hypertension.",
            "Gene therapy offers potential treatments for genetic disorders.",
            "Invasive procedures in medicine may carry risks, including infections.",
            "The blood-brain barrier prevents harmful substances from entering the brain.",
            "Molecular biology is a field of science that focuses on the structure and function of molecules in living organisms.",
            "Radiology plays a key role in diagnosing diseases and conditions through imaging techniques.",
            "Immunotherapy is a form of cancer treatment that uses the body's immune system to fight cancer cells."
        ]
        
        # Embed all the data into ChromaDB
        embed_texts_with_chroma(self.data, CHROMA_COLLECTION_NAME)
        
        # Create document objects for each data entry
        documents = [Document(text=text) for text in self.data]
        
        # Build the index using the documents
        self.index = SimpleKeywordTableIndex(documents)

    def retrieve_context(self, query):
        """
        Retrieve the most relevant context for the given query using ChromaDB.
        """
        collection = self.chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        embedding = np.random.rand(1536)  # Replace with actual query embedding
        results = collection.query(query_embeddings=[embedding.tolist()], n_results=1)
        return results["documents"][0] if results["documents"] else "No relevant context found."

    def calculate_rouge(self, reference, hypothesis):
        """
        Calculate ROUGE scores between reference and hypothesis.
        """
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference, avg=True)
        return scores

    def generate_reasoning_with_gpt(self, context, query):
        """
        Generate reasoning or answer using GPT (OpenAI API).
        """
        try:
            prompt = f"Given the context:\n{context}\nAnswer the following question:\n{query}"
            response = openai.Completion.create(
                engine="gpt-4",  # You can use "gpt-3.5-turbo" or "gpt-4" depending on your subscription
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].text.strip()  # Get the generated reasoning/answer from GPT
        except Exception as e:
            logger.error(f"Error generating response with GPT: {e}")
            return "Error generating reasoning."

    def get_response(self, query):
        """
        Generate a response for the given query by combining retrieval and reasoning.
        """
        try:
            # Retrieve context from ChromaDB
            context = self.retrieve_context(query)
            
            # Generate reasoning/answer using GPT
            reasoning = self.generate_reasoning_with_gpt(context, query)
            
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge(context, query)
            
            # Return the complete response including reasoning, context, and ROUGE scores
            return {
                "reasoning": reasoning,
                "context": context,
                "rouge_scores": rouge_scores,  # Ensure this is included
                "final_answer": f"Context: {context}, Reasoning: {reasoning}"
            }
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "reasoning": "Error occurred",
                "context": "Error occurred",
                "rouge_scores": {},
                "final_answer": f"Error: {e}"
            }
