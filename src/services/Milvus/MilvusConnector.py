import logging  # For tracking connection lifecycle, errors, creation steps

from pymilvus import (  # Official Milvus Python SDK
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


class MilvusConnector:
    # TODO: Store connection details
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.connected = False
        # ? Useful for debugging if needed
        logging.basicConfig(level=logging.INFO)

    # TODO: Establish a connection to Milvus server (if not already connected)
    def connect(self):
        if not self.connected:
            try:
                connections.connect(alias="default", host=self.host, port=self.port)
                self.connected = True
                logging.info(f"Connected to Milvus at {self.host}:{self.port}")
            except Exception as e:
                logging.error(f"Connection to Milvus failed: {e}")
                raise

    # TODO: Disconnect from Milvus server (if already connected)
    def disconnect(self):
        if self.connected:
            try:
                connections.disconnect(alias="default")
                self.connected = False
                logging.info("Disconnected from Milvus.")
            except Exception as e:
                logging.error(f"Oops. Failed to disconnect cleanly: {e}")

    # TODO: Check if a collection exists (helps avoid duplicates)
    def has_collection(self, collection_name: str) -> bool:
        return utility.has_collection(collection_name)

    # TODO: Drop a collection if it exists
    def drop_collection(self, collection_name: str):
        # ! Be careful with this — it deletes the entire collection
        if self.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logging.info(f"Collection '{collection_name}' dropped (deleted).")

    # TODO: Create a new collection with a specific schema
    def create_collection(self, collection_name: str, dimension: int):
        # ? If the collection already exists, skip creation
        if self.has_collection(collection_name):
            logging.info(
                f"Collection '{collection_name}' already exists. Skipping creation."
            )
            return

        """ Define the structure/schema for the collection:
                id: primary key (manual, not auto-generated)
                embedding: the vector field
                text: raw test case or prompt text (for reference)
        """
        fields = [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        ]

        # ? Wrap the fields in a collection schema and create the actual collection
        schema = CollectionSchema(
            fields=fields, description="Test Plan Embedding Collection"
        )
        collection = Collection(name=collection_name, schema=schema)
        logging.info(
            f"Collection '{collection_name}' created with vector dimension {dimension}."
        )

    # TODO: Create an index for the vector in the collection
    def create_index(
        self,
        collection_name: str,
        field_name: str = "embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
    ):
        if not self.has_collection(
            collection_name
        ):  # Make sure the collection exists before indexing
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        collection = Collection(
            name=collection_name
        )  # Load the collection and prepare the index config
        index_params = {
            "index_type": index_type,  # IVF_FLAT is good for moderate-sized datasets
            "params": {"nlist": 128},  # Tune this based on data size
            "metric_type": metric_type,  # COSINE is best for semantic similarity
        }

        # Create the index on the vector field
        collection.create_index(field_name=field_name, index_params=index_params)
        logging.info(
            f"Index created on '{field_name}' using {index_type} with {metric_type} metric."
        )

    # TODO: List all available collections (useful for debugging)
    def list_collections(self):
        try:
            return utility.list_collections()
        except Exception as e:
            logging.error(f"Failed to list collections: {e}")
            return []

    # TODO: Auto-disconnect when object is deleted (if not already disconnected)
    def __del__(self):
        self.disconnect()
