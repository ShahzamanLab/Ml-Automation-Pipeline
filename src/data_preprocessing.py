import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

class Data_preprocessing:
    """
    Data Preprocessing Pipeline.
    All methods accept parameters instead of asking interactively.
    """

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset.copy()

    def remove_duplicates(self) -> pd.DataFrame:
        """Drop duplicate rows and return the cleaned DataFrame."""
        self.dataset = self.dataset.drop_duplicates().reset_index(drop=True)
        return self.dataset

    def handle_missing(
        self,
        method: str = 'impute',
        impute_strategy: str = 'mean',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values.

        Parameters:
            method: 'impute', 'drop', 'ffill', 'bfill'
            impute_strategy: for 'impute' – 'mean', 'median', 'most_frequent', 'constant'
            fill_value: used if impute_strategy == 'constant'
        """
        df = self.dataset.copy()

        if method == 'drop':
            df = df.dropna()
        elif method == 'ffill':
            df = df.ffill()
        elif method == 'bfill':
            df = df.bfill()
        else:  # impute
            for col in df.columns:
                if df[col].isnull().sum() == 0:
                    continue
                # Automatically choose strategy if not specified
                if impute_strategy == 'auto':
                    if df[col].dtype in [np.float64, np.int64]:
                        strategy = 'mean'
                    else:
                        strategy = 'most_frequent'
                else:
                    strategy = impute_strategy

                if strategy == 'constant' and fill_value is not None:
                    imp = SimpleImputer(strategy=strategy, fill_value=fill_value)
                else:
                    imp = SimpleImputer(strategy=strategy)

                df[col] = imp.fit_transform(df[[col]]).ravel()

        self.dataset = df
        return self.dataset

    def encode_categorical(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'onehot',
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical columns.

        Parameters:
            columns: list of column names; if None, encode all object/category columns.
            method: 'onehot' or 'label'
            drop_first: for one-hot, whether to drop the first category.
        """
        df = self.dataset.copy()
        if columns is None:
            columns = list(df.select_dtypes(include=['object', 'category', 'string']).columns)

        if method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:  # onehot
            for col in columns:
                ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
                encoded = ohe.fit_transform(df[[col]])
                col_names = ohe.get_feature_names_out([col])
                encoded_df = pd.DataFrame(encoded, columns=col_names, index=df.index)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

        self.dataset = df
        return self.dataset

    def process_date_column(
        self,
        date_col: str,
        drop_original: bool = True,
        extract_features: List[str] = None
    ) -> pd.DataFrame:
        """
        Extract date features from a datetime column.

        Parameters:
            date_col: name of the column to process.
            drop_original: whether to drop the original column after extraction.
            extract_features: list of features to extract, e.g. ['year','month','day','weekday','quarter'].
        """
        if extract_features is None:
            extract_features = ['year', 'month', 'day', 'weekday', 'quarter']

        df = self.dataset.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        if 'year' in extract_features:
            df[f"{date_col}_Year"] = df[date_col].dt.year
        if 'month' in extract_features:
            df[f"{date_col}_Month"] = df[date_col].dt.month
        if 'day' in extract_features:
            df[f"{date_col}_Day"] = df[date_col].dt.day
        if 'weekday' in extract_features:
            df[f"{date_col}_Weekday"] = df[date_col].dt.day_name()
            # Also one‑hot encode weekday (optional – you may want to do this later)
        if 'quarter' in extract_features:
            df[f"{date_col}_Quarter"] = df[date_col].dt.quarter

        if drop_original:
            df = df.drop(columns=[date_col])

        self.dataset = df
        return self.dataset

    def scale_features(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Scale numerical features.

        Parameters:
            columns: list of columns to scale; if None, scale all numeric columns.
            method: 'standard' or 'minmax'
        """
        df = self.dataset.copy()
        if columns is None:
            columns = list(df.select_dtypes(include=[np.number]).columns)

        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[columns] = scaler.fit_transform(df[columns])
        self.dataset = df
        return self.dataset

    def split_xy(self, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split the dataset into features (X) and target (y)."""
        if target_col not in self.dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        X = self.dataset.drop(columns=[target_col])
        y = self.dataset[target_col]
        return X, y