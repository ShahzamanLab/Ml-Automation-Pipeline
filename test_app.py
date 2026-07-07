import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import zipfile

# Import your backend classes
from src.data_loader import Data_loader
from src.data_preprocessing import Data_preprocessing
from src.feature_engineering import Feature_engineering
from src.model_selection import AutoModelTrainer
from src.evaluate import ModelEvaluation

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS (unchanged) ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Exo+2:wght@300;400;600;800;900&display=swap');
:root {
    --royal: #1A3ADB; --royal-light: #2E52F2; --royal-dark: #0D1F99;
    --black: #060810; --dark: #0B0F1E; --dark2: #111629;
    --white: #FFFFFF; --silver: #B8C4E8; --accent: #4DDFFF;
    --success: #00E676; --warn: #FFB300; --error: #FF3D57;
    --grad1: linear-gradient(135deg, #060810 0%, #0D1F99 50%, #1A3ADB 100%);
    --grad2: linear-gradient(90deg, #1A3ADB, #4DDFFF);
    --grad3: linear-gradient(135deg, #0B0F1E, #111629);
}
html, body, .main, [data-testid="stAppViewContainer"] {
    background: var(--black) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--white) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060810 0%, #0D1B6B 60%, #0B0F1E 100%) !important;
    border-right: 1px solid rgba(26,58,219,0.4) !important;
}
[data-testid="stSidebar"] * { color: var(--white) !important; }
.stButton > button {
    background: var(--grad2) !important; color: var(--black) !important;
    font-family: 'Exo 2', sans-serif !important; font-weight: 800 !important;
    border: none !important; border-radius: 6px !important; padding: 10px 24px !important;
    font-size: 0.95rem !important; letter-spacing: 0.5px !important;
    transition: all 0.3s !important; box-shadow: 0 0 20px rgba(77,223,255,0.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 0 35px rgba(77,223,255,0.5) !important; }
[data-testid="baseButton-primary"] { background: linear-gradient(90deg, #1A3ADB, #00E676) !important; color: #fff !important; font-weight: 900 !important; font-size: 1rem !important; }
.stSelectbox > div, .stMultiSelect > div { background: var(--dark2) !important; color: var(--white) !important; border: 1px solid rgba(26,58,219,0.5) !important; border-radius: 6px !important; }
.stTextInput > div > div, .stNumberInput > div > div { background: var(--dark2) !important; color: var(--white) !important; border: 1px solid rgba(26,58,219,0.5) !important; }
.stSlider > div { background: transparent !important; }
[data-testid="stDataFrameResizable"] { background: var(--dark2) !important; border: 1px solid rgba(26,58,219,0.3) !important; border-radius: 8px !important; }
.stTab [role="tab"] { background: var(--dark2) !important; color: var(--silver) !important; border: 1px solid rgba(26,58,219,0.3) !important; border-radius: 6px 6px 0 0 !important; font-weight: 600 !important; }
.stTab [role="tab"][aria-selected="true"] { background: var(--grad2) !important; color: var(--black) !important; }
.pipeline-hero { background: var(--grad1); border: 1px solid rgba(77,223,255,0.3); border-radius: 16px; padding: 36px 40px; text-align: center; margin-bottom: 32px; position: relative; overflow: hidden; }
.pipeline-hero::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle at 60% 40%, rgba(77,223,255,0.08) 0%, transparent 60%); }
.pipeline-hero h1 { font-family: 'Exo 2', sans-serif; font-size: 2.6rem; font-weight: 900; color: #fff; margin: 0 0 8px 0; letter-spacing: -1px; position: relative; }
.pipeline-hero h1 span { color: var(--accent); }
.pipeline-hero p { color: var(--silver); font-size: 1rem; margin: 0; position: relative; }
.step-card { background: linear-gradient(135deg, #0B0F1E, #111629); border: 1px solid rgba(26,58,219,0.4); border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; position: relative; overflow: hidden; }
.step-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--grad2); border-radius: 12px 0 0 12px; }
.step-card-header { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
.step-number { background: var(--grad2); color: var(--black); font-family: 'Space Mono', monospace; font-weight: 700; font-size: 0.85rem; padding: 6px 14px; border-radius: 20px; letter-spacing: 1px; }
.step-title { font-size: 1.5rem; font-weight: 800; color: #fff; margin: 0; letter-spacing: -0.5px; }
.completed-banner { background: linear-gradient(90deg, rgba(0,230,118,0.15), rgba(26,58,219,0.15)); border: 1px solid rgba(0,230,118,0.4); border-radius: 10px; padding: 16px 22px; display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.completed-banner span { font-size: 1.1rem; font-weight: 700; color: var(--success); }
.info-box { background: rgba(26,58,219,0.15); border: 1px solid rgba(26,58,219,0.4); border-radius: 10px; padding: 18px 22px; color: var(--silver); font-size: 0.95rem; line-height: 1.7; }
.metric-row { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.metric-box { flex: 1; min-width: 130px; background: linear-gradient(135deg, #0D1F99, #1A3ADB); border-radius: 10px; padding: 18px; text-align: center; border: 1px solid rgba(77,223,255,0.2); }
.metric-box .metric-val { font-size: 1.8rem; font-weight: 900; color: var(--accent); font-family: 'Space Mono', monospace; }
.metric-box .metric-label { font-size: 0.75rem; color: var(--silver); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.best-model-card { background: linear-gradient(135deg, #0D4A1A, #0D1F99); border: 2px solid var(--success); border-radius: 14px; padding: 30px; text-align: center; margin: 24px 0; box-shadow: 0 0 40px rgba(0,230,118,0.2); }
.best-model-card h2 { color: var(--success); font-size: 2rem; font-weight: 900; margin: 0 0 8px 0; }
.best-model-card h3 { color: var(--accent); font-size: 1.2rem; font-weight: 600; margin: 0; }
.problem-card { background: linear-gradient(135deg, #0B0F1E, #0D1F99); border: 2px solid rgba(77,223,255,0.3); border-radius: 14px; padding: 24px; text-align: center; cursor: pointer; transition: all 0.3s; margin: 8px; }
.problem-card:hover { border-color: #4DDFFF; box-shadow: 0 0 30px rgba(77,223,255,0.3); transform: translateY(-3px); }
.problem-card.selected { border-color: #00E676; background: linear-gradient(135deg, #0D4A1A, #0D1F99); box-shadow: 0 0 40px rgba(0,230,118,0.2); }
.problem-card h3 { color: #fff; font-size: 1.3rem; font-weight: 800; margin: 8px 0 4px; }
.problem-card p { color: var(--silver); font-size: 0.85rem; margin: 0; }
.folder-card { background: linear-gradient(135deg, #0B0F1E, #111629); border: 1px solid rgba(26,58,219,0.4); border-radius: 12px; padding: 20px; text-align: center; transition: all 0.3s; }
.folder-card:hover { border-color: #FFB300; box-shadow: 0 0 25px rgba(255,179,0,0.2); }
.sidebar-step { padding: 10px 14px; border-radius: 8px; margin: 5px 0; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.sidebar-done { background: rgba(0,230,118,0.15); border: 1px solid rgba(0,230,118,0.4); color: #00E676 !important; }
.sidebar-active { background: rgba(26,58,219,0.3); border: 1px solid rgba(77,223,255,0.5); color: var(--accent) !important; }
.sidebar-pending { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: rgba(255,255,255,0.35) !important; }
.completion-hero { background: var(--grad1); border: 2px solid var(--success); border-radius: 16px; padding: 48px; text-align: center; box-shadow: 0 0 60px rgba(0,230,118,0.15); }
.completion-hero h1 { color: var(--success); font-size: 2.5rem; font-weight: 900; margin: 0 0 10px; }
.completion-hero p { color: var(--silver); font-size: 1.1rem; margin: 0; }
iframe { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
STEPS = [
    'data_loaded',          # 1
    'missing_handled',      # 2
    'features_engineered',  # 3
    'encoded',              # 4
    'problem_type_chosen',  # 5
    'target_selected',      # 6
    'data_split',           # 7
    'models_trained',       # 8
    'evaluated',            # 9
    'models_saved'          # 10 (now download)
]

for s in STEPS:
    if s not in st.session_state:
        st.session_state[s] = False

defaults = {
    'dataset': None,
    'processed_dataset': None,
    'X': None,
    'y': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'target_column': None,
    'problem_type': None,
    'evaluation_results': None,
    'best_model_info': None,
    'trained_models': {},
    'save_folder': None,
    'preprocessor': None,
    'feature_eng': None,
    'evaluator': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

FOLDERS = {
    'regression':       'regression_base_models',
    'classification':   'classification_base_models',
    'time_series':      'time_series_base_models',
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def step_done(name):
    st.session_state[name] = True
    st.rerun()

def sidebar_item(label, done_key, active_condition):
    if st.session_state[done_key]:
        st.sidebar.markdown(f"<div class='sidebar-step sidebar-done'>✅ {label}</div>", unsafe_allow_html=True)
    elif active_condition:
        st.sidebar.markdown(f"<div class='sidebar-step sidebar-active'>⚡ {label}</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<div class='sidebar-step sidebar-pending'>◻️ {label}</div>", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 18px 0 10px;'>
  <div style='font-family:"Space Mono",monospace; font-size:1.4rem; font-weight:700; color:#4DDFFF; letter-spacing:2px;'>AUTO<span style='color:#fff'>ML</span></div>
  <div style='color:rgba(255,255,255,0.4); font-size:0.7rem; letter-spacing:3px; margin-top:2px;'>PIPELINE v4.0</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color:rgba(26,58,219,0.4); margin:8px 0 16px;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:rgba(255,255,255,0.5); font-size:0.7rem; letter-spacing:2px; margin-bottom:8px; padding-left:4px;'>PIPELINE PROGRESS</div>", unsafe_allow_html=True)

sidebar_item("Data Loading",          'data_loaded',          True)
sidebar_item("Missing & Duplicates",  'missing_handled',      st.session_state['data_loaded'])
sidebar_item("Feature Engineering",   'features_engineered',  st.session_state['missing_handled'])
sidebar_item("Categorical Encoding",  'encoded',              st.session_state['features_engineered'])
sidebar_item("Problem Type",          'problem_type_chosen',  st.session_state['encoded'])
sidebar_item("Target Selection",      'target_selected',      st.session_state['problem_type_chosen'])
sidebar_item("Data Splitting",        'data_split',           st.session_state['target_selected'])
sidebar_item("Model Training",        'models_trained',       st.session_state['data_split'])
sidebar_item("Evaluation",            'evaluated',            st.session_state['models_trained'])
sidebar_item("Download Models",       'models_saved',         st.session_state['evaluated'])

done_count = sum(st.session_state[s] for s in STEPS)
pct = int(done_count / len(STEPS) * 100)
st.sidebar.markdown(f"""
<div style='margin:16px 0 8px; padding:0 4px;'>
  <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
    <span style='color:rgba(255,255,255,0.5); font-size:0.7rem; letter-spacing:1px;'>OVERALL PROGRESS</span>
    <span style='color:#4DDFFF; font-family:"Space Mono",monospace; font-size:0.8rem;'>{pct}%</span>
  </div>
  <div style='background:rgba(255,255,255,0.08); border-radius:999px; height:6px;'>
    <div style='background:linear-gradient(90deg,#1A3ADB,#4DDFFF); width:{pct}%; height:6px; border-radius:999px; transition:width 0.5s;'></div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.problem_type:
    pt_display = st.session_state.problem_type.replace('_', ' ').title()
    st.sidebar.markdown(f"<div style='background:rgba(77,223,255,0.1); border:1px solid rgba(77,223,255,0.3); border-radius:8px; padding:10px 14px; margin-top:8px;'><span style='color:rgba(255,255,255,0.5); font-size:0.7rem; letter-spacing:1px;'>PROBLEM TYPE</span><br><span style='color:#4DDFFF; font-weight:800; font-size:1rem;'>{pt_display}</span></div>", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='pipeline-hero'>
  <h1>🤖 <span>Automated</span> ML Pipeline</h1>
  <p>Sequential • Intelligent • Production-Ready — You choose the problem type & models are trained accordingly</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state['data_loaded']:
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 01</span>
        <h2 class='step-title'>📂 Data Loading</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_main, col_info = st.columns([3, 1])
    with col_info:
        st.markdown("""<div class='info-box'>
        <b style='color:#4DDFFF'>Instructions</b><br><br>
        1. Upload a <b>.csv</b> file<br>
        2. Review auto-detected info<br>
        3. Click <b>Complete Step</b>
        </div>""", unsafe_allow_html=True)

    with col_main:
        uploaded = st.file_uploader("Upload CSV Dataset", type="csv", label_visibility="collapsed")
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                loader = Data_loader(tmp_path)
                df = loader.get_dataset()
                st.session_state.dataset = df
                os.unlink(tmp_path)

                mem = df.memory_usage(deep=True).sum() / 1024
                miss_pct = df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100

                st.markdown(f"""
                <div class='metric-row'>
                  <div class='metric-box'><div class='metric-val'>{df.shape[0]:,}</div><div class='metric-label'>Rows</div></div>
                  <div class='metric-box'><div class='metric-val'>{df.shape[1]}</div><div class='metric-label'>Columns</div></div>
                  <div class='metric-box'><div class='metric-val'>{mem:.1f}KB</div><div class='metric-label'>Memory</div></div>
                  <div class='metric-box'><div class='metric-val'>{miss_pct:.1f}%</div><div class='metric-label'>Missing</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(df.head(8), width='stretch')

                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str).values,
                    'Non-Null': df.count().values,
                    'Nulls': df.isnull().sum().values,
                    'Unique': [df[c].nunique() for c in df.columns]
                })
                st.dataframe(dtype_df, width='stretch')

                if st.button("✅ Complete Step 1 — Data Loading", type="primary"):
                    st.session_state.processed_dataset = df.copy()
                    st.session_state.preprocessor = Data_preprocessing(df)
                    step_done('data_loaded')
            except Exception as e:
                st.error(f"Error reading file: {e}")

elif st.session_state['data_loaded']:
    st.markdown("""<div class='completed-banner'><span>✅ Step 1 Complete — Data Loaded</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DUPLICATES & MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['data_loaded'] and not st.session_state['missing_handled']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 02</span>
        <h2 class='step-title'>🔍 Duplicates & Missing Values</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    preprocessor = st.session_state.preprocessor
    if preprocessor is None:
        preprocessor = Data_preprocessing(st.session_state.dataset)
        st.session_state.preprocessor = preprocessor

    with st.expander("🧹 Duplicates", expanded=True):
        dup = preprocessor.dataset.duplicated().sum()
        if dup > 0:
            st.warning(f"⚠️ Found **{dup}** duplicate rows.")
            if st.button("🗑 Remove Duplicates"):
                preprocessor.remove_duplicates()
                st.success("Duplicates removed!")
                st.rerun()
        else:
            st.success("✅ No duplicates found.")

    with st.expander("🕳️ Missing Values", expanded=True):
        miss_df = pd.DataFrame({
            'Column': preprocessor.dataset.columns,
            'Missing': preprocessor.dataset.isnull().sum().values,
            'Missing %': (preprocessor.dataset.isnull().sum().values / len(preprocessor.dataset)*100).round(2)
        })
        miss_df = miss_df[miss_df['Missing'] > 0]
        if not miss_df.empty:
            st.dataframe(miss_df, width='stretch')
            method = st.selectbox("How to handle missing values?",
                ["Impute (mean/median/mode)", "Drop rows", "Forward Fill", "Backward Fill"])
            impute_strategy = "mean"
            if "Impute" in method:
                impute_strategy = st.selectbox("Imputation strategy:", ["mean", "median", "most_frequent"])
            if st.button("▶ Apply Missing Value Strategy"):
                if "Impute" in method:
                    preprocessor.handle_missing(method='impute', impute_strategy=impute_strategy)
                elif "Drop" in method:
                    preprocessor.handle_missing(method='drop')
                elif "Forward" in method:
                    preprocessor.handle_missing(method='ffill')
                else:
                    preprocessor.handle_missing(method='bfill')
                st.success("Missing values handled!")
                st.rerun()
        else:
            st.success("✅ No missing values.")

    if st.button("✅ Complete Step 2 — Duplicates & Missing Handled", type="primary"):
        st.session_state.processed_dataset = preprocessor.dataset
        step_done('missing_handled')

elif st.session_state['missing_handled']:
    st.markdown("""<div class='completed-banner'><span>✅ Step 2 Complete — Duplicates & Missing Handled</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING (drop columns + date extraction)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['missing_handled'] and not st.session_state['features_engineered']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 03</span>
        <h2 class='step-title'>⚙️ Feature Engineering</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_work = st.session_state.processed_dataset
    st.dataframe(df_work.head(8), width='stretch')

    # Drop columns
    drop_cols = st.multiselect("Select columns to drop (e.g., IDs, irrelevant features):", df_work.columns)
    if drop_cols:
        if st.button("🗑 Drop Selected Columns"):
            fe = Feature_engineering(df_work)
            fe.drop_columns(drop_cols)
            st.session_state.processed_dataset = fe.dataset
            st.success(f"Dropped: {drop_cols}")
            st.rerun()

    # Date processing
    with st.expander("📅 Date Column Processing", expanded=True):
        obj_cols = list(st.session_state.processed_dataset.select_dtypes(include=['object']).columns)
        if obj_cols:
            date_col = st.selectbox("Select date column (or skip):", ["— Skip —"] + obj_cols)
            if date_col != "— Skip —":
                if st.button("🗓 Extract Date Features"):
                    preprocessor = Data_preprocessing(st.session_state.processed_dataset)
                    preprocessor.process_date_column(date_col, drop_original=True,
                                                     extract_features=['year','month','day','weekday','quarter'])
                    st.session_state.processed_dataset = preprocessor.dataset
                    st.success(f"Extracted features from {date_col}")
                    st.rerun()
        else:
            st.info("No object columns found for date processing.")

    if st.button("✅ Complete Step 3 — Feature Engineering", type="primary"):
        step_done('features_engineered')

elif st.session_state['features_engineered']:
    st.markdown("""<div class='completed-banner'><span>✅ Step 3 Complete — Feature Engineering Done</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CATEGORICAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['features_engineered'] and not st.session_state['encoded']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 04</span>
        <h2 class='step-title'>🏷️ Categorical Encoding</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_work = st.session_state.processed_dataset
    cat_cols = list(df_work.select_dtypes(include=['object', 'category', 'string']).columns)
    if cat_cols:
        st.info(f"Found categorical columns: {', '.join(cat_cols)}")
        enc_method = st.radio("Encoding method:", ["onehot", "label"], horizontal=True)
        enc_cols = st.multiselect("Select columns to encode:", cat_cols, default=cat_cols)
        if enc_cols and st.button("▶ Apply Encoding"):
            preprocessor = Data_preprocessing(df_work)
            preprocessor.encode_categorical(columns=enc_cols, method=enc_method)
            st.session_state.processed_dataset = preprocessor.dataset
            st.success("Encoding applied!")
            st.rerun()
    else:
        st.success("✅ No categorical columns to encode.")

    if st.button("✅ Complete Step 4 — Categorical Encoding", type="primary"):
        step_done('encoded')

elif st.session_state['encoded']:
    st.markdown("""<div class='completed-banner'><span>✅ Step 4 Complete — Categorical Encoding Done</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PROBLEM TYPE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['encoded'] and not st.session_state['problem_type_chosen']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 05</span>
        <h2 class='step-title'>🧠 Choose Your Problem Type</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    prob_choice = st.radio(
        "Select your problem type:",
        options=["📈 Regression", "🏷️ Classification", "⏱️ Time Series"],
        horizontal=True,
        label_visibility="collapsed"
    )
    desc_map = {
        "📈 Regression": {"label": "Regression", "desc": "Predict continuous numeric value.", "models": ["Linear", "Ridge", "Lasso", "DT", "RF", "GB"]},
        "🏷️ Classification": {"label": "Classification", "desc": "Predict discrete class.", "models": ["Logistic", "DT", "RF", "GB", "KNN", "SVM"]},
        "⏱️ Time Series": {"label": "Time Series", "desc": "Forecast sequential data.", "models": ["ARIMA", "ExpSmoothing"]}
    }
    info = desc_map[prob_choice]
    st.markdown(f"""
    <div style='background:rgba(26,58,219,0.2); border:1px solid rgba(77,223,255,0.4); border-radius:12px; padding:24px; margin:20px 0;'>
      <h3 style='color:#4DDFFF;'>{info['label']}</h3>
      <p style='color:#B8C4E8;'>{info['desc']}</p>
      <div style='color:rgba(255,255,255,0.5); font-size:0.75rem; letter-spacing:1px; text-transform:uppercase;'>Models: {', '.join(info['models'])}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✅ Confirm Problem Type", type="primary"):
        pt_map = {"📈 Regression": "regression", "🏷️ Classification": "classification", "⏱️ Time Series": "time_series"}
        st.session_state.problem_type = pt_map[prob_choice]
        step_done('problem_type_chosen')

elif st.session_state['problem_type_chosen']:
    pt_display = st.session_state.problem_type.replace('_', ' ').title()
    st.markdown(f"""<div class='completed-banner'><span>✅ Step 5 Complete — Problem: <b>{pt_display}</b></span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TARGET SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['problem_type_chosen'] and not st.session_state['target_selected']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 06</span>
        <h2 class='step-title'>🎯 Target Variable</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_work = st.session_state.processed_dataset
    target_col = st.selectbox("Choose the TARGET column:", df_work.columns)
    if target_col:
        st.session_state.target_column = target_col
        td = df_work[target_col]
        if st.session_state.problem_type in ['regression', 'classification']:
            if td.dtype in ['object', 'category'] or td.nunique() <= 20:
                fig = px.bar(x=td.value_counts().index.astype(str), y=td.value_counts().values,
                             title=f"Class Distribution — {target_col}",
                             color_discrete_sequence=['#1A3ADB'])
            else:
                fig = px.histogram(td, title=f"Distribution — {target_col}",
                                   color_discrete_sequence=['#4DDFFF'])
        else:
            fig = px.line(y=td.values, title=f"Time Series — {target_col}",
                          color_discrete_sequence=['#4DDFFF'])
            fig.update_layout(xaxis_title="Index", yaxis_title=target_col)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(11,15,30,0.8)',
            font_color='#B8C4E8', title_font_color='#fff',
            xaxis=dict(gridcolor='rgba(26,58,219,0.2)'),
            yaxis=dict(gridcolor='rgba(26,58,219,0.2)')
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("✅ Complete Step 6 — Target Selected", type="primary"):
        step_done('target_selected')

elif st.session_state['target_selected']:
    st.markdown(f"""<div class='completed-banner'><span>✅ Step 6 Complete — Target: <b>{st.session_state.target_column}</b></span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — DATA SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['target_selected'] and not st.session_state['data_split']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 07</span>
        <h2 class='step-title'>✂️ Train / Test Split</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_work = st.session_state.processed_dataset
    target = st.session_state.target_column
    X = df_work.drop(columns=[target])
    y = df_work[target]

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    with col2:
        rng = st.number_input("Random State", 0, 100, 42, 1,
                              disabled=(st.session_state.problem_type == 'time_series'))

    if st.session_state.problem_type == 'time_series':
        split_idx = int(len(X) * (1 - test_size))
        X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        from sklearn.model_selection import train_test_split
        try:
            strat = y if st.session_state.problem_type == 'classification' else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rng, stratify=strat)
        except:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rng)

    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-box'><div class='metric-val'>{X_tr.shape[0]:,}</div><div class='metric-label'>Train Samples</div></div>
      <div class='metric-box'><div class='metric-val'>{X_te.shape[0]:,}</div><div class='metric-label'>Test Samples</div></div>
      <div class='metric-box'><div class='metric-val'>{X_tr.shape[1]}</div><div class='metric-label'>Features</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✅ Complete Step 7 — Data Split", type="primary"):
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_tr
        st.session_state.X_test = X_te
        st.session_state.y_train = y_tr
        st.session_state.y_test = y_te
        step_done('data_split')

elif st.session_state['data_split']:
    st.markdown(f"""<div class='completed-banner'><span>✅ Step 7 Complete — Train: {st.session_state.X_train.shape[0]:,} | Test: {st.session_state.X_test.shape[0]:,}</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['data_split'] and not st.session_state['models_trained']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 08</span>
        <h2 class='step-title'>🚀 Model Training</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pt = st.session_state.problem_type
    st.markdown(f"<p style='color:#4DDFFF; font-weight:700;'>Problem: {pt.upper()}</p>", unsafe_allow_html=True)

    if st.button("🚀 Train All Models", type="primary"):
        models = {}
        if pt == 'regression':
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            model_map = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
            }
        elif pt == 'classification':
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            model_map = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(random_state=42)
            }
        else:  # time series
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model_map = {}
            y_train_ts = st.session_state.y_train
            try:
                arima = ARIMA(y_train_ts, order=(5,1,0)).fit()
                model_map['ARIMA'] = arima
            except: pass
            try:
                es = ExponentialSmoothing(y_train_ts, trend='add', seasonal=None).fit()
                model_map['ExpSmoothing'] = es
            except: pass

        progress_bar = st.progress(0)
        status = st.empty()
        for i, (name, model) in enumerate(model_map.items()):
            status.markdown(f"Training **{name}**...")
            if pt != 'time_series':
                model.fit(st.session_state.X_train, st.session_state.y_train)
            models[name] = model
            progress_bar.progress((i+1)/len(model_map))

        folder = FOLDERS[pt]
        os.makedirs(folder, exist_ok=True)
        for name, model in models.items():
            with open(os.path.join(folder, f"{name}.pkl"), 'wb') as f:
                pickle.dump(model, f)

        st.session_state.trained_models = models
        st.success(f"✅ {len(models)} models trained and saved to `{folder}/`")
        st.rerun()

    if st.session_state.trained_models:
        if st.button("✅ Complete Step 8 — Models Trained", type="primary"):
            step_done('models_trained')

elif st.session_state['models_trained']:
    st.markdown(f"""<div class='completed-banner'><span>✅ Step 8 Complete — {len(st.session_state.trained_models)} Models Trained</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — EVALUATION (Fixed DataFrame)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['models_trained'] and not st.session_state['evaluated']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 09</span>
        <h2 class='step-title'>📊 Model Evaluation</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pt = st.session_state.problem_type
    folder = FOLDERS[pt]

    if st.button("📊 Evaluate All Models", type="primary"):
        evaluator = ModelEvaluation(model_dir=folder, problem_type=pt)
        evaluator.load_models()
        if pt == 'time_series':
            results = evaluator.evaluate(
                y_test=st.session_state.y_test,
                y_train=st.session_state.y_train
            )
        else:
            results = evaluator.evaluate(
                X_test=st.session_state.X_test,
                y_test=st.session_state.y_test
            )
        evaluator.get_best()
        st.session_state.evaluator = evaluator
        st.session_state.evaluation_results = evaluator.results
        st.session_state.best_model_info = {
            'model': evaluator.best_model_name,
            'score': evaluator.best_score,
            'score_name': 'R²' if pt == 'regression' else 'Accuracy' if pt == 'classification' else 'RMSE'
        }
        st.rerun()

    if st.session_state.evaluation_results:
        evals = st.session_state.evaluation_results
        if pt == 'time_series':
            results_df = pd.DataFrame.from_dict(evals, orient='index').reset_index()
            results_df.columns = ['Model', 'RMSE', 'MAE', 'MAPE']
            st.dataframe(results_df, width='stretch')
            fig = px.bar(results_df, x='Model', y='RMSE', title="RMSE (lower is better)",
                         color='RMSE', color_continuous_scale=['#4DDFFF','#FF3D57'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            results_df = pd.DataFrame(list(evals.items()), columns=['Model', 'Score'])
            st.dataframe(results_df, width='stretch')
            fig = px.bar(results_df, x='Model', y='Score', title="Model Scores",
                         color='Score', color_continuous_scale=['#0D1F99','#4DDFFF'])
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.best_model_info:
            bm = st.session_state.best_model_info
            st.markdown(f"""
            <div class='best-model-card'>
              <h2>🏆 BEST MODEL: {bm['model']}</h2>
              <h3>{bm['score_name']}: {bm['score']:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)

        if st.button("✅ Complete Step 9 — Evaluation Done", type="primary"):
            step_done('evaluated')

elif st.session_state['evaluated']:
    st.markdown("""<div class='completed-banner'><span>✅ Step 9 Complete — Models Evaluated</span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — DOWNLOAD MODELS (No copying, just download)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['evaluated'] and not st.session_state['models_saved']:
    st.markdown("---")
    st.markdown("""
    <div class='step-card'>
      <div class='step-card-header'>
        <span class='step-number'>STEP 10</span>
        <h2 class='step-title'>💾 Download Trained Models</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("All models are saved in the folder during training. You can download them individually or as a ZIP archive.")

    src_folder = FOLDERS[st.session_state.problem_type]
    model_files = [f for f in os.listdir(src_folder) if f.endswith('.pkl')]

    if not model_files:
        st.warning("No model files found in the folder.")
    else:
        st.write(f"Found {len(model_files)} model file(s):")
        for f in model_files:
            st.write(f"- {f}")

        st.subheader("Download Individual Models")
        for f in model_files:
            file_path = os.path.join(src_folder, f)
            with open(file_path, 'rb') as file:
                data = file.read()
            st.download_button(
                label=f"⬇️ Download {f}",
                data=data,
                file_name=f,
                mime="application/octet-stream",
                key=f"download_{f}"
            )

        st.subheader("Download All Models as ZIP")
        if st.button("📦 Create ZIP Archive and Download"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for f in model_files:
                    file_path = os.path.join(src_folder, f)
                    zip_file.write(file_path, arcname=f)
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download all models as ZIP",
                data=zip_buffer,
                file_name=f"models_{st.session_state.problem_type}.zip",
                mime="application/zip",
                key="download_zip"
            )

        if st.button("✅ Complete Step 10 — Models Downloaded", type="primary"):
            st.session_state.save_folder = src_folder
            step_done('models_saved')

elif st.session_state['models_saved']:
    st.markdown(f"""<div class='completed-banner'><span>✅ Step 10 Complete — Models ready to download from <b>{st.session_state.save_folder}/</b></span></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPLETION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['models_saved']:
    st.markdown("---")
    bm = st.session_state.best_model_info
    st.markdown("""
    <div class='completion-hero'>
      <h1>🎉 Pipeline Complete!</h1>
      <p>All steps executed successfully — your models are trained, evaluated, and ready for download.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    ds = st.session_state.processed_dataset
    col1.metric("Dataset Shape", f"{ds.shape[0]} × {ds.shape[1]}")
    col2.metric("Problem Type", st.session_state.problem_type.replace('_', ' ').title())
    col3.metric("Best Model", bm['model'] if bm else "N/A")
    col4.metric("Save Folder", st.session_state.save_folder or "N/A")

    if st.button("🔄 Start New Pipeline"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<hr style='border-color:rgba(26,58,219,0.3); margin:16px 0 10px;'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:rgba(255,255,255,0.2); font-size:0.65rem; text-align:center; letter-spacing:1px;'>AutoML Pipeline v4.0 · User-Driven Problem Type</div>", unsafe_allow_html=True)