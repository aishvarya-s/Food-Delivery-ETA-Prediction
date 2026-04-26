import streamlit as st


st.write("App started successfully")
import pandas as pd

from src.preprocessing.preprocess import load_data, clean_data, handle_missing, remove_outliers, feature_engineering
from src.models.regression_model import run_regression
from src.models.classification_model import run_classification
from src.clustering.clustering import main as run_clustering



# PAGE CONFIG

st.set_page_config(page_title="Delivery ETA Dashboard", layout="wide")

st.title("🚚 Food Delivery ETA Prediction Dashboard")



# LOAD DATA

@st.cache_data
def load_and_process():
    path = "data/raw/Zomato Dataset.csv"

    df = load_data(path)
    df = clean_data(df)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = feature_engineering(df)

    return df


df = load_and_process()

st.success("Data loaded and preprocessed successfully!")


# SIDEBAR

st.sidebar.header("Navigation")

option = st.sidebar.radio(
    "Go to:",
    ["Overview", "Regression (ETA)", "Classification (Delay)", "Clustering"]
)


# OVERVIEW

if option == "Overview":
    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.dataframe(df.head())

    with col2:
        st.write("Summary Stats")
        st.write(df.describe())



# REGRESSION

elif option == "Regression (ETA)":
    st.subheader("ETA Prediction (Regression Models)")

    if st.button("Train Models"):
        trained_models, results = run_regression(df)

        st.success("Models trained successfully!")

        # Show results
        for name, res in results.items():
            st.write(f"### {name}")
            st.write(f"MAE: {res['mae']:.2f}")
            st.write(f"RMSE: {res['rmse']:.2f}")
            st.write(f"R²: {res['r2']:.4f}")



# CLASSIFICATION

elif option == "Classification (Delay)":
    st.subheader("Delay Classification")

    if st.button("Train Classifier"):
        model = run_classification(df)
        st.success("Classification model trained!")



# CLUSTERING

elif option == "Clustering":
    st.subheader("Clustering Analysis")

    if st.button("Run Clustering"):
        fig = run_clustering(df)
        st.pyplot(fig)