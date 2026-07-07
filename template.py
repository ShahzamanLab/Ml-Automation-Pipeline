from pathlib import Path 
import os 


list_of_files = [
    "README.md",
    "requirements.txt",
    ".env",

    "app.py",
    "test_app.py",

    "src/",
    "src/__init__.py",

    "src/utils.py",
    "src/classes_test.ipynb",

    "src/data_loader.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",

    "src/model_selection.py",
    "src/evaluate.py",

    "tests/",
    "tests/__init__.py",
    "tests/test_data_loader.py",
    "tests/test_pipeline.py"
]

for file_path in list_of_files:
    path = Path(file_path)
    folder = path.parent
    if folder != Path("."):
        os.makedirs(folder, exist_ok=True)
    if path.suffix:  
        path.touch(exist_ok=True)