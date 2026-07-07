import pandas as pd
import numpy as np

class Data_loader:
    """this class able to load data right now this class is just work on the csv version"""

    def __init__(self,path:str):
        self.path = path
        try:
            self._dataset = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(f"file not found error: {self.path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
        
    def get_dataset(self) -> pd.DataFrame:
        return self._dataset
