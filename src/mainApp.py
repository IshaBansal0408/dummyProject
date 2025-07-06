from CSVDataInspector import CSVDataInspector
from DataLoaderClass import DataLoaderClass


def main():
    loader = DataLoaderClass()
    loader.uploadFiles()
    loader.convert2CSV()
    loader.combineAllTCs()
    inspector = CSVDataInspector(loader.finalCombinedCSV)
    # inspector.run_full_report()
    inspector.plot_specific_categorical_distributions(
        columns=["functionalarea", "priority"]
    )


if __name__ == "__main__":
    main()
