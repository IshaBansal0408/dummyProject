from sentence_transformers import SentenceTransformer
from pymilvus import Collection

model = SentenceTransformer("all-MiniLM-L6-v2")
collection_name = "testCSVCodeMilvus_trail3"
collection = Collection(name=collection_name)
collection.load()


def search_milvus(query_text):
    query_embedding = model.encode([query_text]).tolist()
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=5,
        output_fields=["Row_text"],
    )
    output_rows = []
    for hit in results[0]:
        original = hit.entity.get("Row_text")
        output_rows.append({"Score": f"{hit.distance:.4f}, Text: {original}"})
    return output_rows


df_out = search_milvus("Enter a query text to search")
puthon.DataFrame(columns=["Score", "Text"])
print(df_out.head())
