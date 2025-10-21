# First, you need to install the necessary library:
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingUtils:
    """
    A utility class to handle the creation of sentence embeddings.
    
    This class uses a singleton pattern to ensure the powerful (and large)
    embedding model is loaded into memory only once, improving performance.
    """
    _model = None
    _model_name = 'all-MiniLM-L6-v2' # A good, fast, and lightweight model

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        """Loads the sentence transformer model into memory."""
        if cls._model is None:
            print(f"Loading embedding model '{cls._model_name}' into memory...")
            cls._model = SentenceTransformer(cls._model_name)
            print("Embedding model loaded successfully.")
        return cls._model

    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        """
        Generates a vector embedding for a given piece of text.

        Args:
            text (str): The input text to be converted into an embedding.

        Returns:
            List[float]: The vector embedding as a list of floats.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
            
        model = cls._get_model()
        embedding = model.encode(text, convert_to_tensor=False)
        
        # Ensure the output is a standard Python list of floats
        return embedding.tolist()

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing the Embedding Utility ---")
    
    # The model will be downloaded and loaded on the first call
    text1 = "John helped me with a Python script last week."
    embedding1 = EmbeddingUtils.generate_embedding(text1)
    
    print(f"\nText 1: '{text1}'")
    print(f"Generated embedding (first 5 dimensions): {embedding1[:5]}")
    print(f"Embedding dimensions: {len(embedding1)}") # Should be 384 for this model

    # The model is already in memory, so this call is much faster
    text2 = "Who was it that assisted me with that coding problem?"
    embedding2 = EmbeddingUtils.generate_embedding(text2)

    print(f"\nText 2: '{text2}'")
    print(f"Generated embedding (first 5 dimensions): {embedding2[:5]}")
    
    # Calculate cosine similarity to show that the embeddings are semantically close
    # (This is what the vector database does automatically)
    embedding1_np = np.array(embedding1)
    embedding2_np = np.array(embedding2)
    similarity = np.dot(embedding1_np, embedding2_np) / (np.linalg.norm(embedding1_np) * np.linalg.norm(embedding2_np))
    
    print(f"\nCosine Similarity between Text 1 and Text 2: {similarity:.4f}")
    print("A high similarity score (close to 1.0) shows the model understands the meaning.")
