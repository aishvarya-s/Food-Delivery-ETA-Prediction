from preprocessing.preprocess import (
    load_data, clean_data, handle_missing,
    remove_outliers, feature_engineering
)

from models.classification_model import run_classification
from models.regression_model import run_regression

def main():
    path = "data/raw/Zomato Dataset.csv"

    df = load_data(path)
    print("Data loaded successfully.")
    df = clean_data(df)
    print("Data cleaned successfully.")
    df = handle_missing(df)
    print("Missing values handled successfully.")
    df = remove_outliers(df)
    print("Outliers removed successfully.")
    df = feature_engineering(df)
    print("Feature engineering completed successfully.")

    print("Data after preprocessing:")
    print(df.shape)
    print(df.head())
    print(df.info())
    print("Before Classification:")
    run_classification(df)
    print("Classification model trained and evaluated successfully.")

    trained_models, results = run_regression(df)

if __name__ == "__main__":
    main()
