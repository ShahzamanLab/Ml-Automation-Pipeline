import streamlit as st
import pandas as pd
import os
import numpy as np
from src.data_loader import Data_loader
from src.data_preprocessing import Data_preprocessing
from src.feature_engineering import Feature_engineering
from src.model_selection import AutoModelTrainer
from src.evaluate import ModelEvaluation



path = r"D:\Automation_pipeline\archive (4).zip" 
extracted_path = r"D:\Automation_pipeline\src"

#with zipfile.ZipFile(path, "r") as zip_ref:
    #zip_ref.extractall(extracted_path)

data_pth = r"D:\Automation_pipeline\src\stock_data.csv"
loader = Data_loader(data_pth)
df = loader.get_dataset()

# ---------------- Preprocessing ----------------
preprocess = Data_preprocessing(df)

preprocess.check_duplicates()
preprocess.check_missing_values()
preprocess.Date_Column_handler()

df = preprocess.Categorical_Value_dealer()

# ---------------- Feature Engineering ----------------
feature = Feature_engineering(df)
df = feature.feature_remover()  # Use real column name

# ---------------- Split X and y ----------------
preprocess = Data_preprocessing(df)
X, y = preprocess.split_xy()

print(df.head(3))
print(df.shape)

trainer = AutoModelTrainer(X, y)
trainer.set_test_params()         # Ask user for test_size & random_state
trainer.choose_problem_type()     # Ask user problem type
trainer.train_models()            # Train all models for that problem
evaluator = ModelEvaluation()

evaluator.choose_model_type()   # choose regression/classification/time_series
evaluator.load_models()         # load all .pkl models from folder

# evaluate using test data
evaluator.evaluate(X_test=trainer.X_test, y_test=trainer.y_test)

# Optional: predict using best model
predictions = evaluator.predict(trainer.X_test)
print(predictions[:5])           