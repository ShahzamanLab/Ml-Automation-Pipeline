# AutoML Pipeline – Industrial‑Grade Automated Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Scikit‑learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18.0-blueviolet)](https://plotly.com/)

---

## 📌 Overview

**AutoML Pipeline** is a **fully interactive, end‑to‑end machine learning platform** built with [Streamlit](https://streamlit.io/). It guides you through the entire ML lifecycle – from raw CSV upload to trained, evaluated, and downloadable models – with a **modern, responsive UI** and **modular backend classes** that can be reused in other applications.

Whether you are a **data scientist** looking for a quick prototyping tool, a **business analyst** wanting to experiment with ML without writing code, or a **developer** integrating ML into your product, this pipeline provides a **transparent, controllable** environment that puts you in the driver’s seat.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📂 Data Loading** | Upload CSV files; automatic preview of rows, columns, memory usage, missing values. |
| **🧹 Duplicate & Missing Handling** | Detect and remove duplicates; handle missing values via imputation, dropping, or filling. |
| **⚙️ Feature Engineering** | Drop irrelevant columns; extract date features (year, month, day, weekday, quarter). |
| **🏷️ Categorical Encoding** | One‑hot or label encoding for categorical columns – choose exactly which columns to transform. |
| **🧠 Problem Selection** | Pick between **Regression**, **Classification**, or **Time Series** – models are tailored accordingly. |
| **🎯 Target Selection** | Visualise target distribution (bar, histogram, line) and confirm your dependent variable. |
| **✂️ Train/Test Split** | Set test size and random state; automatic stratification for classification, sequential split for time series. |
| **🚀 Model Training** | Train a suite of state‑of‑the‑art models with progress feedback: <br> *Regression:* Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting. <br> *Classification:* Logistic, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM. <br> *Time Series:* ARIMA and Exponential Smoothing (Holt‑Winters). |
| **📊 Model Evaluation** | Compute and display performance metrics (R², Accuracy, RMSE, MAE, MAPE) with interactive bar charts. Automatically selects the **best model**. |
| **💾 Model Download** | Download each model individually as a `.pkl` file, or all models as a **single ZIP archive** – no file‑copy conflicts. |
| **📈 Pipeline Progress** | Sidebar tracks your progress through 10 steps with a visual indicator and percentage. |
| **🔄 Reset** | One‑click restart to begin a new pipeline with a fresh dataset. |
| **🎨 Modern UI** | Custom dark theme with neon accents, card‑based layout, and responsive design. |

---

## 🏗️ Architecture

The application is built on a **clean separation of concerns**:



AutoML-Pipeline/
├── app.py # Streamlit frontend (UI logic)
├── src/
│ ├── data_loader.py # Loads CSV and provides basic info
│ ├── data_preprocessing.py # Duplicates, missing, encoding, date, scaling
│ ├── feature_engineering.py# Drop columns, more feature ops
│ ├── model_selection.py # Trains regression/classification/time series models
│ └── evaluate.py # Evaluates saved models, picks the best
├── README.md
└── requirements.txt




### Backend Classes

| Class | Responsibility |
|-------|----------------|
| `Data_loader` | Load CSV from a path, return pandas DataFrame, provide dataset info. |
| `Data_preprocessing` | Remove duplicates; handle missing values (impute/drop/fill); encode categorical columns (label/one‑hot); process date columns; scale features (standard/min‑max). All methods accept explicit parameters – no interactive prompts. |
| `Feature_engineering` | Drop any set of columns; extendable for additional transformations (e.g., polynomial features, interactions). |
| `AutoModelTrainer` | Accepts `X`, `y` and a problem type; splits data (or uses pre‑split sets); trains the appropriate model suite; saves models to disk. |
| `ModelEvaluation` | Loads models from a folder; evaluates on test data; stores all scores; returns the best model by the primary metric (R², accuracy, or RMSE). |

All classes are **frontend‑agnostic** – they can be used in a Jupyter notebook, a Flask API, or any other Python environment.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShahzamanLab/Ml-Automation-Pipeline
   cd automl-pipeline

2.  **Create a virtual environment (recommended)**
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3.  **Install dependencies**
    pip install -r requirements.txt

4. **Launch the app**
    streamlit run app.py



📖 Usage Walkthrough
The pipeline consists of 10 sequential steps. Each step must be completed before moving to the next – this ensures that you don’t miss any critical preprocessing or configuration.

Step 1: Data Loading
Upload a CSV file.

Review the auto‑generated summary and preview.

Click the Complete Step 1 button to lock in your dataset.

Step 2: Duplicates & Missing Values
Check for duplicates and remove them if necessary.

Inspect missing values per column.

Choose a strategy (impute, drop, fill) and apply it.

Step 3: Feature Engineering
Drop any columns you don’t want (e.g., IDs, constants).

If you have a date column, extract features like year, month, day, weekday, and quarter.

Step 4: Categorical Encoding
See which columns are categorical.

Select which to encode and choose between One‑Hot or Label encoding.

Step 5: Problem Type
Choose Regression, Classification, or Time Series.

A summary of the models that will be trained is displayed.

Step 6: Target Selection
Pick the column you want to predict.

A visualisation helps you understand its distribution.

Step 7: Train/Test Split
Set test size and random state.

For time series, the split is sequential; for classification, it’s stratified.

Step 8: Model Training
Click the Train All Models button.

Watch the progress bar as each model is fitted.

All models are automatically saved in a dedicated folder.

Step 9: Evaluation
Click Evaluate All Models.

See a table of scores and a bar chart.

The best model is highlighted.

Step 10: Download Models
Download individual models or a ZIP containing all of them.

Finish
The final dashboard summarises your pipeline and allows you to start over.

🧪 Example Use Cases
Predicting house prices – upload your housing dataset, treat it as regression, and find the best regressor.

Customer churn prediction – classification, view class balance, and get the most accurate classifier.

Stock price forecasting – time series, use ARIMA or Exponential Smoothing to forecast future values.

Rapid prototyping – quickly test multiple models on any tabular dataset without writing a single line of code.

🔧 Customisation
The backend classes are designed to be extensible. You can:

Add new models in model_selection.py by extending the model_map dictionaries.

Implement custom feature engineering methods in feature_engineering.py.

Change the evaluation metrics in evaluate.py to use, for example, F1‑score or MSE.

Add a prediction pipeline that loads the best model and makes predictions on new data – the PredictPipeline class (shown in the class definitions) is ready for this.

🤝 Contributing
We welcome contributions! If you have an idea for a new feature, a bug fix, or improved documentation, please:

Fork the repository.

Create a feature branch (git checkout -b feature/amazing-feature).

Commit your changes (git commit -m 'Add amazing feature').

Push to the branch (git push origin feature/amazing-feature).

Open a Pull Request.

Please ensure your code follows the existing style and includes appropriate tests (where applicable).

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgements
Streamlit – for making data apps so easy.

Scikit‑learn – the backbone of our ML models.

Plotly – for beautiful interactive charts.

Statsmodels – for time series functionality.

📧 Contact
For questions, suggestions, or collaboration, reach out to:
Your Name – your.email@example.com

