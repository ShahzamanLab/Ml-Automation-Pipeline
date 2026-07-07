from src.data_loader import Data_loader
from src.data_preprocessing import Data_preprocessing
from src.feature_engineering import Feature_engineering
from src.model_selection import AutoModelTrainer
from src.evaluate import ModelEvaluation
import pandas as pd

## Data Loder class testing
data_path = r"D:\Automation_pipeline\Ml-Automation-Pipeline\src\stock_data.csv"
loader = Data_loader(data_path)
dataset = loader.get_dataset()
#print(dataset.head(10))

## Data Preporcessing class Testing

preprocessor = Data_preprocessing(dataset=dataset)
preprocessor.check_duplicates()
preprocessor.check_missing_values()
preprocessor.Date_Column_handler()
