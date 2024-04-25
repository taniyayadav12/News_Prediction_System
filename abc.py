import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Collection
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Step 2: Data Preprocessing
def preprocess_data(data):
    data = data.dropna()  # Remove rows with missing values
    return data

# Step 3: Feature Selection and Engineering (For simplicity, we'll use only 'Close' price as the feature)
def select_features(data):
    features = data[['Close']]
    return features

# Step 4: Model Training
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)

# Step 6: Make Predictions
def make_predictions(model, data):
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='B')  # Predicting for the next 30 business days
    future_data = pd.DataFrame(index=future_dates, columns=data.columns)
    future_data['Close'] = model.predict(future_data.index.to_numpy().reshape(-1, 1))
    return future_data

# Step 7: Visualization
def visualize_results(data, future_data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Stock Price')
    plt.plot(future_data.index, future_data['Close'], label='Predicted Stock Price', linestyle='--')
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Function
if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    stock_symbol = 'AAPL'

    # Step 1: Data Collection
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

    # Step 2: Data Preprocessing
    preprocessed_data = preprocess_data(stock_data)

    # Step 3: Feature Selection and Engineering
    features = select_features(preprocessed_data)
    target = preprocessed_data['Close']

    # Step 4: Model Training
    model, X_test, y_test = train_model(features, target)

    # Step 5: Model Evaluation
    evaluate_model(model, X_test, y_test)

    # Step 6: Make Predictions
    future_data = make_predictions(model, preprocessed_data)

    # Step 7: Visualization
    visualize_results(preprocessed_data, future_data)
