# HuggingFace-based library optimized for sentence-level embeddings.
from sentence_transformers import SentenceTransformer
import logging


class MilvusEmbedder:
    # TODO: Loads the SentenceTransformer model at init where Default model: all-MiniLM-L6-v2
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            # ? This model gives 384-dim embeddings, good for general similarity tasks
            self.model = SentenceTransformer(model_name)
            logging.info(f"Embedder initialized with model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load embedding model '{model_name}': {e}")
            raise

    # TODO: Generate vector embeddings from a list of input texts
    def encode(self, texts: list, batch_size: int = 32, normalize: bool = True):
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings.")

        try:
            # ? normalize_embeddings=True scales vectors to unit length (optional but good for cosine similarity)
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=normalize,
            )
            logging.info(f"Successfully encoded {len(texts)} texts.")
            return embeddings
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            raise
