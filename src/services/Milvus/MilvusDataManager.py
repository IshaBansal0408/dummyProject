from pymilvus import Collection
import logging


class MilvusDataManager:
    def __init__(self, connector, collection_name):
        self.connector = connector
        self.collection_name = collection_name

    def insert_embeddings(self, ids, embeddings, texts):
        if not (len(ids) == len(embeddings) == len(texts)):
            raise ValueError("Length mismatch among ids, embeddings, and texts.")

        collection = Collection(self.collection_name)
        entities = [ids, embeddings, texts]
        collection.insert(entities)
        logging.info(
            f"Inserted {len(ids)} vectors into collection '{self.collection_name}'."
        )

    def batch_insert_embeddings(self, ids, embeddings, texts, batch_size=500):
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            logging.info(f"Inserting records {start + 1} to {end}")
            batch_ids = ids[start:end]
            batch_emb = embeddings[start:end]
            batch_texts = texts[start:end]
            self.insert_embeddings(batch_ids, batch_emb, batch_texts)
        logging.info("Batch insertion completed.")

    def search(self, query_embedding, top_k=5):
        collection = Collection(self.collection_name)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        results = collection.search(
            [query_embedding],
            "embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"],
        )
        res_list = []
        for hits in results:
            for hit in hits:
                res_list.append(
                    {
                        "id": hit.id,
                        "score": hit.distance,
                        "text": hit.entity.get("text"),
                    }
                )
        return res_list
