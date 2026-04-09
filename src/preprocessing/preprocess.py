import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# load dataset
def load_data(path):
    df = pd.read_csv(path)
    return df


# basic cleaning
def clean_data(df):
    df = df.copy()

    # fix column names (remove spaces and brackets)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # clean target column
    df['Time_taken_min'] = df['Time_taken_min'].astype(str)
    df['Time_taken_min'] = df['Time_taken_min'].str.extract(r'(\d+)')
    df['Time_taken_min'] = pd.to_numeric(df['Time_taken_min'], errors='coerce')

    return df


# handle missing values
def handle_missing(df):
    df = df.copy()

    # numerical columns
    num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # categorical columns
    cat_cols = [
        'Weather_conditions',
        'Road_traffic_density',
        'Festival',
        'City',
        'Type_of_vehicle'
    ]

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# remove extreme delivery times
def remove_outliers(df):
    df = df.copy()

    Q1 = df['Time_taken_min'].quantile(0.25)
    Q3 = df['Time_taken_min'].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df['Time_taken_min'] >= Q1 - 1.5 * IQR) &
        (df['Time_taken_min'] <= Q3 + 1.5 * IQR)
    ]

    return df


# calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians,
                                [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# create useful features
def feature_engineering(df):
    df = df.copy()

    # distance feature
    df['distance'] = haversine(
        df['Restaurant_latitude'],
        df['Restaurant_longitude'],
        df['Delivery_location_latitude'],
        df['Delivery_location_longitude']
    )

    # convert order time
    df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce')

    # extract hour
    df['order_hour'] = df['Time_Orderd'].dt.hour

    # mark peak hours
    df['is_peak'] = df['order_hour'].apply(
        lambda x: 1 if x in [12, 13, 19, 20, 21] else 0
    )

    return df


# encode categorical data (not used yet, for later)
def encode_data(df):
    df = df.copy()
    df = pd.get_dummies(df, drop_first=True)
    return df


# scale numerical features (not used yet)
def scale_features(df):
    df = df.copy()

    scaler = StandardScaler()

    num_cols = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'distance'
    ]

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


# main pipeline
def main():
    path = "data/raw/Zomato Dataset.csv"

    df = load_data(path)
    df = clean_data(df)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = feature_engineering(df)

    print("final shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()