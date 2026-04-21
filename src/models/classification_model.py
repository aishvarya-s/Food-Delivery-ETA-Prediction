import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# create target variable
def create_target(df):
    def classify_delay(x):
        if x <= 0:
            return "On-Time"
        elif x <= 10:
            return "Slight Delay"
        else:
            return "High Delay"

    df['delay_class'] = df['delay'].apply(classify_delay)
    return df


# prepare features
def prepare_data(df):
    X = df.drop([
        'Time_taken_min',
        'delay_class',
        'Time_Orderd',
        'ID',
        'Delivery_person_ID',
        'Restaurant_latitude',
        'Restaurant_longitude',
        'Delivery_location_latitude',
        'Delivery_location_longitude',
        'delay',
        'expected_time'
        
    ], axis=1)

    y = df['delay_class']

    X = pd.get_dummies(X, drop_first=True)

    X = X.fillna(0)

    return X, y


# train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# pipeline function
def run_classification(df):
    df = create_target(df)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    return model