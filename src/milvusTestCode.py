from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

connections.connect(host="127.0.0.1", port="19530")
print("Connected to Milvus successfully!")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")

collection_name = "testEmbeddingMilvus"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Existing collection '{collection_name}' dropped!")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
]

schema = CollectionSchema(
    fields, description="Store single texts with embedding values"
)
collection = Collection(name=collection_name, schema=schema)
print("Collection created successfully!")

texts = [
    "Milvus is a vector database.",
    "Pinecone and Weaviate are also vector databases.",
    "Sentence transformers convert text to vectors.",
    "I like working with embeddings and search systems.",
    "This is a test sentence for Milvus search.",
]

vectors = model.encode(texts).tolist()

print("Embeddings formed are as follows:")
for i, (text, vector) in enumerate(zip(texts, vectors)):
    print(f"\n[{i + 1}] Text: {text}")
    print(f"Embedding (dim={len(vector)}):")
    print(vector[:10], "...")

entities = [vectors, texts]
collection.insert(entities)
print(f" Inserted {len(texts)} texts successfully!")

collection.flush()

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    },
)
print("Index created successfully!")

collection.load()
print("Collection loaded into memory successfully!")

query_text = "This is a test query to check Milvus Database"
query_vector = model.encode(query_text).tolist()

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text"],
)

print(f"\n Topmost results for: '{query_text}'")
for hit in results[0]:
    print(f"- {hit.entity.get('text')} (score: {hit.distance:.4f})")
