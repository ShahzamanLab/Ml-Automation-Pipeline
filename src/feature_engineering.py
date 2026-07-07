import pandas as pd
from typing import List, Optional

class Feature_engineering:
    """Feature engineering operations (drop columns, create interactions, etc.)"""

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy()

    def drop_columns(self, columns: List[str]) -> pd.DataFrame:
        """Drop the specified columns from the dataset."""
        invalid = [c for c in columns if c not in self.dataset.columns]
        if invalid:
            raise ValueError(f"Columns not found: {invalid}")
        self.dataset = self.dataset.drop(columns=columns)
        return self.dataset

    # You can add more feature engineering methods here (e.g., create interaction terms, polynomial features, etc.)