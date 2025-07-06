import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# TODO: Inspect and summarize CSV data
class CSVDataInspector:
    # TODO: Initialize with a CSV file path or a pandas DataFrame
    def __init__(self, data):
        """
        Args: data (str or pd.DataFrame): Path to CSV file or a DataFrame.
        """
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"CSV file not found: {data}")
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Input must be a file path or a pandas DataFrame.")

    # TODO: Provide a summary of the DataFrame
    def overview(self):
        print("\n📊 Data Overview")
        print("-" * 40)
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

    # TODO: Print data types and non-null counts
    def info(self):
        print("\n🧠 Data Types & Non-null Info")
        print("-" * 40)
        print(self.df.info())

    # TODO: Show descriptive statistics for numeric columns
    def describe(self):
        print("\n📈 Descriptive Statistics")
        print("-" * 40)
        print(self.df.describe())

    # TODO: Count and percentage of missing values per column
    def missing_values(self):
        print("\n❗ Missing Values")
        print("-" * 40)
        null_counts = self.df.isnull().sum()
        null_perc = (null_counts / len(self.df)) * 100
        missing_df = pd.DataFrame(
            {"Missing Count": null_counts, "Missing %": null_perc}
        ).sort_values("Missing Count", ascending=False)
        print(missing_df[missing_df["Missing Count"] > 0])

    # TODO: Show top N unique values per categorical column
    def unique_counts(self, top_n=5):
        print("\n🔢 Unique Value Counts (Top categories)")
        print("-" * 40)
        for col in self.df.select_dtypes(include="object").columns:
            print(f"\nColumn: {col}")
            print(self.df[col].value_counts(dropna=False).head(top_n))

    # TODO: Run all reports in sequence
    def run_full_report(self):
        self.overview()
        self.info()
        self.describe()
        self.missing_values()
        self.unique_counts()

    # TODO: Plot distributions of specified categorical columns side-by-side in one figure.
    def plot_specific_categorical_distributions(self, columns=None):
        """Args: columns (list): List of column names to plot."""
        n = len(columns)
        fig, axes = plt.subplots(1, n, figsize=(10 * n, 10))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, columns):
            if col not in self.df.columns:
                ax.text(
                    0.5, 0.5, f"Column '{col}' not found.", ha="center", va="center"
                )
                ax.axis("off")
                continue

            value_counts = self.df[col].value_counts(dropna=False)
            sns.barplot(
                x=value_counts.index,
                y=value_counts.values,
                palette="muted",
                edgecolor="black",
                ax=ax,
            )
            for patch in ax.patches:
                height = patch.get_height()
                ax.annotate(
                    f"{int(height)}",
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title(f"Distribution of '{col}'", fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis="x", rotation=90)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()
