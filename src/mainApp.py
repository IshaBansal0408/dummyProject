# For logging, file discovery, and path ops
import logging
import glob
import os

# Custom Modules Import
from dataLoaders.CSVDataInspector import CSVDataInspector
from dataLoaders.CSVLoaderClass import CSVDataLoader
from dataLoaders.DataLoaderClass import DataLoaderClass
from services.Milvus.MilvusConnector import MilvusConnector
from services.Milvus.MilvusDataManager import MilvusDataManager
from services.Milvus.MilvusEmbedder import MilvusEmbedder
from services.Milvus.MilvusSearchCLI import MilvusSearchCLI


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting Milvus Embedding Pipeline...")

    try:
        logging.info("Step 1: Uploading and processing Excel files...")
        dataLoader = DataLoaderClass()
        dataLoader.uploadFiles()
        dataLoader.convert2CSV()
        dataLoader.combineAllTCs()
        logging.info("Excel files loaded and converted successfully.")

        logging.info("Step 2: Running CSV inspection and plots...")
        inspector = CSVDataInspector(dataLoader.finalCombinedCSV)
        inspector.run_full_report()
        inspector.plot_specific_categorical_distributions(
            columns=["functionalarea", "priority"]
        )
        logging.info("CSV inspection completed.")

        logging.info("Step 3: Loading processed CSV...")
        files = sorted(glob.glob("../data/processed/combined_testcases_*.csv"))
        if not files:
            raise FileNotFoundError("No combined testcases CSV found!")
        latest_file = files[-1]

        csvLoader = CSVDataLoader(latest_file)
        csvLoader.load_csv()
        csvLoader.clean_data()
        descriptions = csvLoader.extract_text_column("description")
        df_metadata = csvLoader.df.copy()

        if "id" not in df_metadata.columns:
            df_metadata.reset_index(inplace=True)
            df_metadata.rename(columns={"index": "id"}, inplace=True)
        else:
            df_metadata["id"] = df_metadata["id"].astype(int)

        filtered = [(i, d) for i, d in enumerate(descriptions) if d and d.strip()]
        if not filtered:
            raise ValueError("No valid descriptions found after cleaning!")
        ids, descriptions = zip(*filtered)
        ids = [int(i) + 1 for i in ids]
        logging.info(f"Loaded {len(ids)} valid descriptions.")

        logging.info("Step 4: Connecting to Milvus...")
        connector = MilvusConnector()
        connector.connect()
        logging.info("Connected to Milvus.")

        COLLECTION_NAME = "test_plan_embeddings"
        EMBEDDING_DIM = 384

        if not connector.has_collection(COLLECTION_NAME):
            logging.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
            connector.create_collection(COLLECTION_NAME, EMBEDDING_DIM)
            connector.create_index(COLLECTION_NAME)
            logging.info(f"Collection '{COLLECTION_NAME}' created and indexed.")
        else:
            logging.info(f"Collection '{COLLECTION_NAME}' already exists.")

        embedder = MilvusEmbedder()
        data_manager = MilvusDataManager(connector, COLLECTION_NAME)

        logging.info("Step 5: Generating and inserting embeddings in batches...")

        BATCH_SIZE = 16  # Use smaller batch size to prevent memory overload
        total = len(descriptions)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_descriptions = descriptions[start:end]
            batch_ids = ids[start:end]

            try:
                logging.info(f"Encoding batch {start} to {end}...")
                batch_embeddings = embedder.encode(
                    batch_descriptions, batch_size=BATCH_SIZE
                )
                logging.info(f"Inserting batch {start} to {end} into Milvus...")
                data_manager.batch_insert_embeddings(
                    batch_ids, batch_embeddings, batch_descriptions
                )
                logging.info(f"Batch {start} to {end} inserted successfully.")
            except Exception as e:
                logging.error(f"Error processing batch {start}-{end}: {e}")

        logging.info("All embeddings inserted.")

        logging.info("Step 6: Launching interactive CLI...")
        cli = MilvusSearchCLI(embedder, data_manager, df_metadata=df_metadata)
        cli.interactive_cli()

        logging.info("Shutting down...")
        connector.disconnect()
        logging.info("Disconnected from Milvus. Pipeline finished.")

    except Exception as e:
        logging.exception(f"Pipeline failed due to error: {e}")
