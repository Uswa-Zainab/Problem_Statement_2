import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Model MSE: {mse}")
    return mse

def save_model(model, model_out):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to: {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with processed CSV files")
    parser.add_argument("--model_out", required=True, help="Output path for trained model")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, args.model_out)
