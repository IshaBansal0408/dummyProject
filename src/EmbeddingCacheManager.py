import logging
import os

import pandas as pd


class EmbeddingCacheManager:
    """Handles saving/loading embeddings + metadata cache as CSV."""

    def __init__(self, cache_path):
        self.cache_path = cache_path

    def save(self, df):
        df.to_csv(self.cache_path, index=False)
        logging.info(f"Saved embeddings cache to {self.cache_path}")

    def load(self):
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")
        df = pd.read_csv(self.cache_path)
        # Parse embedding string back to list of floats
        df["embedding"] = df["embedding"].apply(
            lambda x: list(map(float, x.split(",")))
        )
        logging.info(f"Loaded embeddings cache from {self.cache_path}")
        return df
