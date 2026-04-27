import streamlit as st




st.write("App started successfully")
import pandas as pd


from src.preprocessing.preprocess import load_data, clean_data, handle_missing, remove_outliers, feature_engineering
from src.models.regression_model import run_regression, predict_single_eta
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
   ["Overview", "Regression (ETA)", "Classification (Delay)", "Clustering", "ETA Simulator"]
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
      
elif option == "ETA Simulator":
   st.subheader("ETA Prediction Simulator")


   # user inputs
   distance = st.number_input("Distance (km)", min_value=0.0)


   weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"])
   traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
   order_type = st.selectbox("Order Type", ["Snack", "Meal", "Drinks", "Buffet"])
   vehicle_type = st.selectbox("Vehicle Type", ["Motorcycle", "Scooter", "Electric Scooter"])
   city = st.selectbox("City", ["Urban", "Metropolitian", "Semi-Urban"])
   festival = st.selectbox("Festival", ["Yes", "No"])


   if st.button("Predict ETA"):
       sample = {
           'Delivery_person_Age': 29,
           'Delivery_person_Ratings': 4.5,
           'distance': distance,
           'Weather_conditions': weather,
           'Road_traffic_density': traffic,
           'Vehicle_condition': 2,
           'Type_of_order': order_type,
           'Type_of_vehicle': vehicle_type,
           'multiple_deliveries': 0,
           'Festival': festival,
           'City': city
       }


       trained_models, _ = run_regression(df)
       eta = predict_single_eta(trained_models, sample)


       st.success(f"Predicted ETA: {eta} minutes")