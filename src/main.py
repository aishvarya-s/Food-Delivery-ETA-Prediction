import os
from preprocessing.preprocess import (
    load_data, clean_data, handle_missing,
    remove_outliers, feature_engineering
)

from models.classification_model import run_classification
from models.regression_model import run_regression
from models.regression_model import run_regression, predict_single_eta
from clustering.clustering import main as run_clustering


def get_user_input():
    print("\n--- ETA Simulator ---")
    print("Enter delivery details:\n")

    distance = float(input("Distance (in km)                              : "))
    
    print("Weather conditions: Sunny / Cloudy / Fog / Stormy / Sandstorms / Windy")
    weather = input("Weather                                       : ")

    print("Traffic density: Low / Medium / High / Jam")
    traffic = input("Traffic                                       : ")

    print("Type of order: Snack / Meal / Drinks / Buffet")
    order_type = input("Type of order                                 : ")

    print("Type of vehicle: motorcycle / scooter / electric_scooter")
    vehicle_type = input("Type of vehicle                               : ")

    print("City: Urban / Metropolitian / Semi-Urban")
    city = input("City                                          : ")

    print("Festival: Yes / No")
    festival = input("Festival                                      : ")

    sample = {
        'Delivery_person_Age'    : 29,      # default median
        'Delivery_person_Ratings': 4.5,     # default median
        'distance'               : distance,
        'Weather_conditions'     : weather,
        'Road_traffic_density'   : traffic,
        'Vehicle_condition'      : 2,       # default median
        'Type_of_order'          : order_type,
        'Type_of_vehicle'        : vehicle_type,
        'multiple_deliveries'    : 0,       # default
        'Festival'               : festival,
        'City'                   : city
    }

    return sample


def main():
    path = "data/raw/Zomato Dataset.csv"

    # --- load and preprocess data ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "data", "raw", "Zomato Dataset.csv")

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
    #print(df['distance'].describe())

    print("Data after preprocessing:")
    print(df.shape)
    print("\nData after preprocessing:")
    print(f"Shape: {df.shape}")
    print(df.head())
    print(df.info())
    print("Before Classification:")

    # --- classification ---
    print("\n--- Running Classification ---")
    run_classification(df)
    print("Classification model trained and evaluated successfully.")
    print("Classification completed successfully.")

    # --- regression ---
    print("\n--- Running Regression ---")
    trained_models, results = run_regression(df)
    print("Running clustering analysis...")
    print("Regression completed successfully.")

    # --- clustering ---
    print("\n--- Running Clustering ---")
    run_clustering(df)
    print("Clustering completed.")
    print("Clustering completed successfully.")

    # --- dynamic ETA simulation ---
    while True:
        sample = get_user_input()
        predicted_eta = predict_single_eta(trained_models, sample)
        print(f"\nPredicted ETA: {predicted_eta} minutes")

        again = input("\nSimulate another delivery? (yes / no): ")
        if again.strip().lower() != 'yes':
            print("Exiting ETA Simulator.")
            break


if __name__ == "__main__":
    main()