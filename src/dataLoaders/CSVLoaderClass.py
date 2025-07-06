import logging  # Logs status/errors instead of printing
import os

import pandas as pd


class CSVDataLoader:
    # TODO: Initialize with the full path of the CSV file
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        # ? Quick check to ensure the file exists at the provided path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found at: {filepath}")

        logging.info(f"CSVDataLoader initialized with file: {filepath}")

    # TODO: Load CSV into a pandas DataFrame
    def load_csv(self):
        try:
            self.df = pd.read_csv(self.filepath)
            logging.info(f"CSV loaded successfully with shape: {self.df.shape}")
        except Exception as e:
            logging.error(f"Failed to load CSV: {e}")
            raise

    # TODO: Clean data — drop empty rows, trim strings, etc.
    def clean_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")

        # Drop rows where all elements are NaN
        self.df.dropna(how="all", inplace=True)

        # Strip leading/trailing spaces from string columns only
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = self.df[col].map(
                lambda x: x.strip() if isinstance(x, str) else x
            )

        logging.info("Data cleaned: NaNs dropped, strings stripped.")

    # TODO: Extract specific text columns (like 'description' for embedding)
    def extract_text_column(self, column_name: str) -> list:
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")

        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV.")

        # Drop rows where the column is empty/null and return as list of strings
        texts = self.df[column_name].dropna().astype(str).tolist()
        logging.info(
            f"Extracted {len(texts)} non-null entries from column '{column_name}'"
        )
        return texts

    # TODO: (Optional) Return the whole DataFrame for inspection
    def get_dataframe(self):
        return self.df
