import pymongo
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np
import pandas as pd
import mlflow
import os

mlflow.set_tracking_uri('https://lanaben:7ddc36c3f8a57b0df9f619f50e6f0a2cf6da5f55@dagshub.com/lanaben/MBajk-prediction-app.mlflow')
mlflow.set_experiment('Bike Stand Predictions Metrics')

client = MongoClient('mongodb+srv://lanabenedicic:lana@cluster0.zmxuv9j.mongodb.net/')
db = client['requests']
requests_collection = db['requests']

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')
data_path = os.path.join(DATA_DIR, "processed_bike_data", "DVORANA_TABOR_processed_bike_data.csv")
historical_data = pd.read_csv(data_path)
historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])

current_date = datetime.now().date()
start_date = datetime.combine(current_date, datetime.min.time())
end_date = datetime.combine(current_date, datetime.max.time())

last_prediction = requests_collection.find_one(
    {'timestamp': {'$gte': start_date, '$lte': end_date}},
    sort=[('timestamp', pymongo.DESCENDING)]
)

if last_prediction:
    pred_timestamp = last_prediction['timestamp']
    pred_list = last_prediction['prediction']

    predicted_values = []
    actual_values = []

    for pred_val, pred_time in zip(pred_list, pd.date_range(start=pred_timestamp, periods=len(pred_list), freq='H')):
        closest_row = historical_data.iloc[(historical_data['datetime'] - pred_time).abs().argsort()[:1]]

        if not closest_row.empty:
            actual_val = closest_row['available_bike_stands'].values[0]
            predicted_values.append(pred_val)
            actual_values.append(actual_val)

    print(predicted_values)
    print(actual_values)

    if predicted_values and actual_values:
        errors = np.array(actual_values) - np.array(predicted_values)
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors**2)
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")

        with mlflow.start_run():
            mlflow.log_metric("Mean Absolute Error", mae)
            mlflow.log_metric("Mean Squared Error", mse)
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            mlflow.end_run()
else:
    print("No predictions found for today.")