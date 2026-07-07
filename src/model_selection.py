import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Time Series (optional – will be imported only if available)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


class AutoModelTrainer:
    """
    Train a set of models based on the problem type.
    Stores the train/test split and trained models.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,          # 'regression', 'classification', 'time_series'
        test_size: float = 0.2,
        random_state: int = 42,
        save_folder: Optional[str] = None
    ):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        self.save_folder = save_folder

        # Will be set during training
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}          # trained model objects
        self.model_names = []     # list of names

    def _ensure_folder(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        return folder

    def train_models(self) -> Dict[str, Any]:
        """Train all models for the chosen problem type and return them."""
        if self.problem_type == 'regression':
            self._train_regression()
        elif self.problem_type == 'classification':
            self._train_classification()
        elif self.problem_type == 'time_series':
            self._train_time_series()
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

        # Optionally save models if save_folder is given
        if self.save_folder is not None:
            folder = self._ensure_folder(self.save_folder)
            for name, model in self.models.items():
                path = os.path.join(folder, f"{name.replace(' ', '_')}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(model, f)

        return self.models

    def _train_regression(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        model_map = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=self.random_state),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=self.random_state)
        }

        for name, model in model_map.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            self.model_names.append(name)

    def _train_classification(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y if self.y.nunique() > 1 else None
        )

        model_map = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=self.random_state),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=self.random_state),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'SVC': SVC(random_state=self.random_state)
        }

        for name, model in model_map.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            self.model_names.append(name)

    def _train_time_series(self):
        # For univariate time series, we use the target series only.
        # We assume self.y is a pandas Series with datetime index or numeric index.
        series = self.y
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        # Split sequentially (no shuffle)
        split_idx = int(len(series) * (1 - self.test_size))
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]

        # Store as DataFrame/Series for compatibility
        self.y_train = train_series
        self.y_test = test_series
        self.X_train = self.X.iloc[:split_idx] if self.X is not None else None
        self.X_test = self.X.iloc[split_idx:] if self.X is not None else None

        if not _STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for time series models. Install with: pip install statsmodels")

        # ARIMA
        try:
            arima_model = ARIMA(train_series, order=(5, 1, 0))
            arima_fit = arima_model.fit()
            self.models['ARIMA'] = arima_fit
            self.model_names.append('ARIMA')
        except Exception as e:
            print(f"ARIMA training failed: {e}")

        # Exponential Smoothing (Holt-Winters)
        try:
            es_model = ExponentialSmoothing(train_series, trend='add', seasonal=None)
            es_fit = es_model.fit()
            self.models['ExpSmoothing'] = es_fit
            self.model_names.append('ExpSmoothing')
        except Exception as e:
            print(f"Exponential Smoothing training failed: {e}")

        # Note: You can add more TS models or allow hyperparameter tuning later.

    def load_models(self, folder: str) -> Dict[str, Any]:
        """Load pre‑trained models from a folder (for evaluation without retraining)."""
        self.models = {}
        for fname in os.listdir(folder):
            if fname.endswith('.pkl'):
                path = os.path.join(folder, fname)
                with open(path, 'rb') as f:
                    name = fname.replace('.pkl', '')
                    self.models[name] = pickle.load(f)
                    self.model_names.append(name)
        return self.models