from datetime import datetime
import re
from flask_cors import CORS
import glob
import json
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import os
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from pymongo import MongoClient
from unidecode import unidecode

client = MongoClient('mongodb+srv://lanabenedicic:lana@cluster0.zmxuv9j.mongodb.net/')
db = client['requests']
requests_collection = db['requests']

app = Flask(__name__)
CORS(app)
mlflow.set_tracking_uri('https://lanaben:7ddc36c3f8a57b0df9f619f50e6f0a2cf6da5f55@dagshub.com/lanaben/MBajk-prediction-app.mlflow')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'bikes')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

WINDOW_SIZE = 7
NUM_FEATURES = 5
models = {}


def preload_models():
    client = MlflowClient()
    registered_models = client.search_registered_models()

    for model_info in registered_models:
        for version_info in model_info.latest_versions:
            model_name = model_info.name
            model_uri = f"models:/{model_name}/1"

            model = mlflow.pyfunc.load_model(model_uri)

            models[model_name] = model
            print(f"Loaded {model_name} into memory.")
    print(f"Loaded {len(models)} models.")


def return_stations():
    data_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.json'))
    print(RAW_DATA_DIR)
    print(data_files)
    stations = []

    for file_path in data_files:
        try:
            with open(file_path, 'r') as file:
                station = json.load(file)
                if 'name' in station:
                    stations.append(station['name'])
                    print(station['name'])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    if stations:
        return {'Stations': stations}
    else:
        return {'Stations': 'No data found'}


def get_last_values(station_name, x):
    file_name = station_name.replace(" ", "_") + "_processed_data.csv"
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)

    try:
        data = pd.read_csv(file_path, usecols=['available_bike_stands', 'temperature', 'relative_humidity', 'apparent_temperature', 'rain'])

        if x > len(data):
            print("Number of values exceeds the number of available entries. Returning all available entries.")
            x = len(data)

        last_values = data.tail(x)

        flattened_values = last_values.values.flatten().tolist()
        return flattened_values
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


@app.route('/mbajk/stations', methods=['GET'])
def get_stations():
    try:
        response = return_stations()
        if response['Stations'] != 'No data found':
            return jsonify(response)
        else:
            return "No data found", 404
    except Exception as e:
        return f"An error occurred: {e}", 500


@app.route('/mbajk/<name>/<int:limit>', methods=['GET'])
def get_station_data(name, limit):
    try:
        normalized_name = unidecode(name)

        flattened_values = get_last_values(normalized_name, limit)

        if flattened_values:
            return jsonify({"data": flattened_values})
        else:
            return jsonify({"message": "No data found"}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


@app.route('/mbajk/predict/<station_name>', methods=['POST'])
def predict(station_name):
    if not request.is_json:
        return jsonify({"error": "Invalid JSON format"}), 400

    data = request.json
    if "data" not in data:
        return jsonify({"error": "Missing 'data' key in JSON"}), 400

    normalized_station_name = normalize_station_name(station_name)

    scaler = load_scaler(normalized_station_name)
    if scaler is None:
        return jsonify({"error": "Scaler for the specified station could not be loaded"}), 404

    num_features = len(scaler.scale_)
    print(num_features)
    model = load_model(normalized_station_name)
    if model is None:
        return jsonify({"error": "Model for the specified station could not be loaded"}), 404
    data_values = data["data"]
    try:
        data_array = np.array(data_values).reshape(-1, num_features)
        data_normalized = scaler.transform(data_array)
        print(f"data_normalized shape: {data_normalized.shape}")
        data_reshaped = data_normalized.reshape(1, 7, 5)
        print(f"data_reshaped shape: {data_reshaped.shape}")
    except Exception as e:
        return jsonify({"error": f"Error during data preparation: {e}"}), 400

    try:
        prediction_normalized = model.predict(data_reshaped)
        dummy_features = np.zeros((prediction_normalized.shape[1], num_features - 1))
        prediction_expanded = np.hstack([prediction_normalized.T, dummy_features])

        prediction_real = scaler.inverse_transform(prediction_expanded)[:, 0]

        prediction_list = prediction_real.tolist()

        data_to_store = {
            'station_name': station_name,
            'data': data['data'],
            'timestamp': datetime.utcnow(),
            'prediction': prediction_list
        }
        requests_collection.insert_one(data_to_store)

        return jsonify({"prediction": prediction_list})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400



def normalize_station_name(name):
    name = name.upper()
    name = re.sub(r'[\.-]', '', name)
    name = re.sub(r'\s+', '_', name)
    full_model_name = f"train_{name}_processed_data"
    return full_model_name



def load_model(station_name):
    client = MlflowClient()
    try:
        model_info = client.get_registered_model(station_name)
        latest_version = model_info.latest_versions[0].version
        model_uri = f"models:/{station_name}/{latest_version}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model for station {station_name}: {e}")
        return None


def load_scaler(station_name):
    client = MlflowClient()
    try:
        model_versions = client.search_model_versions(f"name='{station_name}'")
        if not model_versions:
            print(f"No versions found for model '{station_name}'.")
            return None

        model_version = sorted(model_versions, key=lambda x: int(x.version))[-1]

        scaler_path = f"{station_name}_scaler.pkl"
        artifacts_path = client.download_artifacts(model_version.run_id, scaler_path)

        scaler = joblib.load(artifacts_path)
        return scaler
    except Exception as e:
        print(f"Error loading scaler for station {station_name}: {e}")
        return None



if __name__ == '__main__':
    print("Starting model preload...")
    preload_models()
    print("Model preload complete.")
    app.run(host='0.0.0.0', port=5000)
