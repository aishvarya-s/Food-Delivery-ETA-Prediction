import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# prepare features and target
def prepare_data(df):

    features = [
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'distance',
        'Weather_conditions',
        'Road_traffic_density',
        'Vehicle_condition',
        'Type_of_order',
        'Type_of_vehicle',
        'multiple_deliveries',
        'Festival',
        'City',
        'is_peak'
    ]

    X = df[features].copy()
    y = df['Time_taken_min']

    X = pd.get_dummies(X, drop_first=True)  # convert categorical to 0s and 1s
    X = X.fillna(0)                          # safety net for missing values

    return X, y



def train_models(X_train, y_train):

    models = {
        'Linear Regression': LinearRegression(),

        'Decision Tree': DecisionTreeRegressor(
            random_state=42,
            max_depth=10,           # stops it memorizing every path
            min_samples_leaf=10,    # each leaf needs at least 10 samples
            min_samples_split=20    # needs 20 samples to even try a split
        ),

        'Random Forest': RandomForestRegressor(
            random_state=42,
            n_estimators=200,       # more trees = more stable
            max_depth=15,           # caps tree depth
            min_samples_leaf=5,     # regularizes individual trees
            max_features='sqrt',    # each tree sees √n features → reduces correlation
            n_jobs=-1               # use all CPU cores
        )
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"{name} trained ")

    return trained


def evaluate_models(trained_models, X_train, X_test, y_train, y_test):

    results = {}

    for name, model in trained_models.items():
        y_pred   = model.predict(X_test)

        mae      = mean_absolute_error(y_test, y_pred)
        rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2  = r2_score(y_test, y_pred)
        train_r2 = r2_score(y_train, model.predict(X_train))
        gap      = train_r2 - test_r2

        
        if gap < 0.05:
            overfit_status = "No overfitting "
        elif gap < 0.15:
            overfit_status = "Slight overfitting "
        else:
            overfit_status = "Overfitting "

        results[name] = {
            'model'   : model,
            'mae'     : mae,
            'rmse'    : rmse,
            'r2'      : test_r2,
            'y_pred'  : y_pred
        }

        print(f"\n{name}")
        print(f"  MAE      : {mae:.2f} mins")
        print(f"  RMSE     : {rmse:.2f} mins")
        print(f"  Train R² : {train_r2:.4f}")
        print(f"  Test  R² : {test_r2:.4f}")
        print(f"  Gap      : {gap:.4f}  ->  {overfit_status}")

    return results


# predict ETA for a single sample
def predict_eta(model, sample):
    prediction = model.predict(sample)
    print(f"\nPredicted Delivery Time: {prediction[0]:.1f} minutes")
    return prediction[0]


# main pipeline — call this from main.py
def run_regression(df):
    print("Starting Regression Pipeline...")

    # step 1 - prepare data
    X, y = prepare_data(df)
    print(f"Features shape: {X.shape}")

    # step 2 - split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]} rows")
    print(f"Test size : {X_test.shape[0]} rows")

    # step 3 - train all models
    trained_models = train_models(X_train, y_train)

    # step 4 - evaluate and check overfitting
    results = evaluate_models(trained_models, X_train, X_test, y_train, y_test)

    # step 5 - pick best model
    best_name  = max(results, key=lambda k: results[k]['r2'])
    best_model = trained_models[best_name]
    print(f"\nBest Model: {best_name} (R² = {results[best_name]['r2']:.4f})")

    # step 6 - show test predictions
    y_pred_all = best_model.predict(X_test)

    prediction_df = pd.DataFrame({
        'Actual_Time (min)'   : y_test.values,
        'Predicted_Time (min)': y_pred_all.round(1),
        'Difference (min)'    : abs(y_test.values - y_pred_all).round(1)
    })

    print("\nTest Data - Sample Predictions (first 10 rows):")
    print(prediction_df.head(10))
    print(f"\nTotal predictions made: {len(prediction_df)}")

    # step 7 - show train predictions (to verify overfitting visually)
    '''y_pred_train = best_model.predict(X_train)

    train_df = pd.DataFrame({
        'Actual_Time (min)'   : y_train.values,
        'Predicted_Time (min)': y_pred_train.round(1),
        'Difference (min)'    : abs(y_train.values - y_pred_train).round(1)
    })

    print("\nTrain Data - Sample Predictions (first 10 rows):")
    print(train_df.head(10)) '''

    return trained_models, results


# On average, our model's prediction is off by ± 3.37 minutes from the actual delivery time based on
# the MAE's value 