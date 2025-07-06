import os  # For path handling and folder creation
from tkinter import Tk  # Provides the GUI to open a file dialog
from tkinter.filedialog import askopenfilenames

import pandas as pd  # Used for reading and manipulating Excel files.

from src.services.HelperClass import HelperClass  # User Defined Class

helper = HelperClass()


class DataLoaderClass:
    def __init__(self):
        self.excelFiles = []  # List of selected Excel files
        self.excelFilesData = {}  # Dict of ExcelFile objects keyed by filename
        self.convertedCSVFileData = {}  # Dict of CSV dataframes keyed by CSV filename
        self.finalCombinedCSV = None  # Final combined dataframe

    # TODO: Uploads Excel files and loads them into pandas ExcelFile objects.
    def uploadFiles(self):
        Tk().withdraw()  # Hide the main Tkinter window
        filepaths = askopenfilenames(
            title="Select Excel Test Plan Files",
            filetypes=[("Excel files", "*.xls *.xlsx")],
        )
        self.excelFiles = list(filepaths)

        for fname in self.excelFiles:
            try:
                self.excelFilesData[fname] = pd.ExcelFile(fname)
            except Exception as e:
                print(f"Failed to load {fname}: {e}")

        if not self.excelFiles:
            print("No Excel files were selected.")

    # TODO: Converts sheets containing test case ID columns into cleaned CSV files.
    def convert2CSV(self):
        for fname, xl in self.excelFilesData.items():
            combined_df_list = []
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                if helper.isTCIDPresent(df.columns):
                    cleaned_columns = [
                        helper.cleanColumnName(col) for col in df.columns
                    ]
                    df.columns = cleaned_columns
                    combined_df_list.append(df)

            if combined_df_list:
                combined_df = pd.concat(combined_df_list, ignore_index=True)
                # No intermediate CSV saving here
                self.convertedCSVFileData[fname] = (
                    combined_df  # Use original Excel filename as key
                )
                print(f"Combined data for: {fname}")
            else:
                print(f"No sheets with valid test case ID columns found in {fname}")

    # TODO: Combines all converted CSV dataframes into one final dataframe.
    def combineAllTCs(self, save_folder: str = "../data/processed"):
        combinedDF_list = list(self.convertedCSVFileData.values())
        if combinedDF_list:
            combinedDF = pd.concat(combinedDF_list, ignore_index=True)
            self.finalCombinedCSV = combinedDF
            os.makedirs(save_folder, exist_ok=True)
            timestamp = helper.get_timestamp()
            filename = f"combined_testcases_{timestamp}.csv"
            save_path = os.path.join(save_folder, filename)

            try:
                combinedDF.to_csv(save_path, index=False)
                full_path = os.path.abspath(save_path)  # Get absolute path
                print(f"Final combined CSV saved at: {full_path}")
            except Exception as e:
                print(f"Error saving final combined CSV: {e}")
        else:
            print("No converted CSV data to combine.")
