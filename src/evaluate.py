import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error


class ModelEvaluation:
    """
    Evaluate trained models on test data.
    Works for regression, classification, and time series.
    """

    def __init__(self, model_dir: str, problem_type: str):
        """
        Args:
            model_dir: path to folder containing .pkl model files.
            problem_type: 'regression', 'classification', or 'time_series'.
        """
        self.model_dir = model_dir
        self.problem_type = problem_type
        self.models = {}          # loaded models
        self.results = {}         # evaluation results per model
        self.best_model = None
        self.best_score = None
        self.best_model_name = None

    def load_models(self) -> Dict[str, Any]:
        """Load all .pkl models from the directory."""
        self.models = {}
        for fname in os.listdir(self.model_dir):
            if fname.endswith('.pkl'):
                path = os.path.join(self.model_dir, fname)
                with open(path, 'rb') as f:
                    name = fname.replace('.pkl', '')
                    self.models[name] = pickle.load(f)
        return self.models

    def evaluate(
        self,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        y_train: Optional[pd.Series] = None   # needed for time series forecasting
    ) -> Dict[str, float]:
        """
        Evaluate each loaded model.

        For regression: R² score (higher is better).
        For classification: accuracy (higher is better).
        For time series: RMSE (lower is better) – but we store RMSE; the best will be the lowest RMSE.

        Returns a dict of model_name -> score (for regression/classification) or dict with RMSE etc.
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")

        self.results = {}

        if self.problem_type in ['regression', 'classification']:
            if X_test is None or y_test is None:
                raise ValueError("X_test and y_test are required for regression/classification evaluation.")

            for name, model in self.models.items():
                try:
                    y_pred = model.predict(X_test)
                    if self.problem_type == 'regression':
                        score = r2_score(y_test, y_pred)
                    else:  # classification
                        score = accuracy_score(y_test, y_pred)
                    self.results[name] = score
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
                    self.results[name] = None

        else:  # time_series
            if y_test is None:
                raise ValueError("y_test is required for time series evaluation.")
            if y_train is None:
                # If not provided, we assume we can use the model's internal data – but better to pass it.
                print("Warning: y_train not provided. Some models may not forecast correctly.")

            for name, model in self.models.items():
                try:
                    # Determine the forecast method
                    if hasattr(model, 'forecast'):
                        # ARIMA and ExpSmoothing have forecast(steps)
                        pred = model.forecast(steps=len(y_test))
                    elif hasattr(model, 'predict'):
                        # Generic predict – need start and end.
                        # If we have y_train, we can set start = len(y_train)
                        if y_train is not None:
                            start = len(y_train)
                            end = len(y_train) + len(y_test) - 1
                            pred = model.predict(start=start, end=end)
                        else:
                            # Fallback: assume predict takes X_test (not typical for TS)
                            pred = model.predict(X_test)
                    else:
                        print(f"Model {name} has no forecast/predict method. Skipping.")
                        continue

                    # Compute error metrics
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    mae = mean_absolute_error(y_test, pred)
                    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100  # avoid division by zero
                    self.results[name] = {
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape
                    }
                except Exception as e:
                    print(f"Error evaluating {name}: {e}")
                    self.results[name] = None

        return self.results

    def get_best(self, metric: str = 'auto') -> tuple:
        """
        Determine the best model based on the evaluation results.

        For regression/classification, the highest score is best.
        For time series, we use RMSE (lowest is best).

        Returns (best_model_name, best_score).
        """
        if not self.results:
            raise ValueError("No evaluation results. Run evaluate() first.")

        if self.problem_type in ['regression', 'classification']:
            # Filter out None values
            valid = {k: v for k, v in self.results.items() if v is not None}
            if not valid:
                raise ValueError("No valid scores found.")
            # Higher is better
            self.best_model_name = max(valid, key=valid.get)
            self.best_score = valid[self.best_model_name]
        else:  # time_series
            # results are dicts with RMSE, MAE, MAPE; we use RMSE
            valid = {k: v for k, v in self.results.items() if v is not None and 'RMSE' in v}
            if not valid:
                raise ValueError("No valid time series results found.")
            # Lower RMSE is better
            self.best_model_name = min(valid, key=lambda k: valid[k]['RMSE'])
            self.best_score = valid[self.best_model_name]['RMSE']

        self.best_model = self.models[self.best_model_name]
        return self.best_model_name, self.best_score

    def predict(self, X: Union[pd.DataFrame, int]) -> np.ndarray:
        """
        Use the best model to make predictions.

        For regression/classification: X should be a DataFrame of features.
        For time series: X should be the number of steps to forecast (int).
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run get_best() first.")

        if self.problem_type == 'time_series':
            if not isinstance(X, int):
                raise ValueError("For time series, X must be an integer (number of steps to forecast).")
            if hasattr(self.best_model, 'forecast'):
                return self.best_model.forecast(steps=X)
            elif hasattr(self.best_model, 'predict'):
                # Need start; we can store training length in the evaluator.
                # We'll assume we have self.training_len set somewhere.
                # For simplicity, we'll use start=0 (may not be accurate).
                # Better to pass y_train during evaluation to know the length.
                raise NotImplementedError("Need to know training length for predict().")
        else:
            return self.best_model.predict(X)