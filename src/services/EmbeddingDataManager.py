import pandas as pd
import os
import logging
from MilvusEmbedder import MilvusEmbedder
from MilvusDataManager import MilvusDataManager
from MilvusConnector import MilvusConnector


class EmbeddingDataManager:
    def __init__(
        self,
        embedder: MilvusEmbedder,
        connector: MilvusConnector = None,
        collection_name: str = None,
    ):
        self.embedder = embedder
        self.connector = connector
        self.collection_name = collection_name
        self.data_manager = None

        if self.connector and self.collection_name:
            # Initialize MilvusDataManager if Milvus integration is desired
            self.data_manager = MilvusDataManager(self.connector, self.collection_name)

    def generate_embeddings(self, texts: list, batch_size: int = 32):
        if not texts:
            raise ValueError("Empty text list provided for embedding.")
        logging.info(f"Generating embeddings for {len(texts)} texts.")
        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        return embeddings

    def save_to_csv(self, descriptions: list, embeddings: list, filepath: str):
        if len(descriptions) != len(embeddings):
            raise ValueError("Descriptions and embeddings length mismatch.")

        # Serialize embeddings as comma-separated strings
        embedding_strings = [",".join(map(str, emb)) for emb in embeddings]

        df = pd.DataFrame({"description": descriptions, "embedding": embedding_strings})

        df.to_csv(filepath, index=False)
        logging.info(f"Saved {len(descriptions)} embeddings to CSV at '{filepath}'.")

    def load_from_csv(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        df = pd.read_csv(filepath)
        if "description" not in df.columns or "embedding" not in df.columns:
            raise ValueError(
                "CSV missing required columns 'description' and/or 'embedding'."
            )

        # Parse embeddings from string back to float lists
        df["embedding"] = df["embedding"].apply(
            lambda x: list(map(float, x.split(",")))
        )
        descriptions = df["description"].tolist()
        embeddings = df["embedding"].tolist()

        logging.info(f"Loaded {len(descriptions)} embeddings from CSV '{filepath}'.")
        return descriptions, embeddings

    def insert_into_milvus(self, descriptions: list, embeddings: list):
        if not self.data_manager:
            raise RuntimeError(
                "MilvusDataManager not initialized. Provide connector and collection_name."
            )

        ids = list(range(1, len(descriptions) + 1))
        result = self.data_manager.insert_embeddings(ids, embeddings, descriptions)
        logging.info(
            f"Inserted {len(ids)} records into Milvus collection '{self.collection_name}'."
        )
        return result
