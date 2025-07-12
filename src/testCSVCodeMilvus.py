import pandas as pd

df = pd.read_csv("../data/processed/sample.csv")
print(df.head())


def row2Text(row):
    return ". ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])


df["row2Text"] = df.apply(row2Text, axis=1)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["row2Text"].tolist(), show_progress_bar=True).tolist()

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

connections.connect(host="127.0.0.1", port="19530")
print("Connected to Milvus successfully!")

collection_name = "testCSVCodeMilvus_trail3"


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="Row_text", dtype=DataType.VARCHAR, max_length=20000),
]
schema = CollectionSchema(
    fields, description="Store single texts with embedding values"
)
if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
else:
    collection = Collection(name=collection_name, schema=schema)


if not collection.has_index():
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )

text = df["row2Text"].tolist()
entities = [embeddings, text]
collection.insert(entities)
collection.flush()
collection.load()

query_text = input("Enter a query text to search: ")
query_embedding = model.encode([query_text]).tolist()
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5,
    output_fields=["Row_text"],
)
for hit in results[0]:
    print(f"Row text: {hit.entity.get('Row_text')}, Distance: {hit.distance}")
print("Search completed successfully!")
