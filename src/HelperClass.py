import re
from datetime import datetime


class HelperClass:
    def __init__(self):
        pass

    # TODO: Converts column name to lowercase and strips non-alphanumeric characters.
    def cleanColumnName(self, column_name: str) -> str:
        """
        Args: column_name (str): Original column name.
        Returns: str: Normalized column name.
        """
        return re.sub(r"[^a-z0-9]", "", column_name.lower())

    # TODO: Checks if any of the columns likely represents a Test Case ID.
    def isTCIDPresent(self, columns: list[str]) -> bool:
        """Args: columns (list[str]): List of column names.
        Returns:bool: True if 'tcid' is detected after normalization.
        """
        cleaned_cols = [self.cleanColumnName(col) for col in columns]
        return "tcid" in cleaned_cols

    # TODO: Returns the current timestamp formatted as a string.
    def get_timestamp(self, fmt: str = "%Y%m%d_%H%M%S") -> str:
        """
        Args: fmt (str): Format string for datetime.strftime. Default: 'YYYYMMDD_HHMMSS'
        Returns: str: Formatted timestamp string.
        """
        return datetime.now().strftime(fmt)
