"""
Singleton-cached embedding utility.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80MB)
Used by the retrieval engine for semantic fallback search and by consolidation
to attach content_vector to each new memory node.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from memory.observability import observer


class EmbeddingUtils:
    """Singleton-cached embedding model loader."""

    _model = None
    _model_name = "all-MiniLM-L6-v2"

    @classmethod
    def _get_model(cls) -> SentenceTransformer:
        """Loads the sentence transformer model into memory (once)."""
        if cls._model is None:
            observer.info("loading embedding model", model=cls._model_name)
            cls._model = SentenceTransformer(cls._model_name)
            observer.info("embedding model ready")
        return cls._model

    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        """Generate a 384-dim embedding vector for `text`."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        model = cls._get_model()
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
