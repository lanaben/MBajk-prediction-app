from flask_cors import CORS
import glob
import json
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import os
from unidecode import unidecode

app = Flask(__name__)
CORS(app)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'bikes')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

WINDOW_SIZE = 7
NUM_FEATURES = 5


def return_stations():
    data_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.json'))
    stations = []

    for file_path in data_files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                stations.extend(data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return {'Stations': stations}


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
        if response['Stations']:
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

    data_values = data["data"]

    scaler = load_scaler(station_name)

    if scaler is None:
        return jsonify({"error": "Scaler for the specified station could not be loaded"}), 404

    num_features = len(scaler.scale_)

    model = load_model(station_name)
    if model is None:
        return jsonify({"error": "Model for the specified station could not be loaded"}), 404

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
        return jsonify({"prediction": prediction_list})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400


def load_model(station_name):
    station_name = unidecode(station_name)
    model_file = os.path.join(MODELS_DIR, station_name.replace(" ", "_") + "_processed_data_model.h5")
    if os.path.exists(model_file):
        return tf.keras.models.load_model(model_file)
    else:
        return None


def load_scaler(station_name):
    station_name = unidecode(station_name)
    scaler_path = os.path.join(ROOT_DIR, 'models', 'scalers')
    scaler_file = os.path.join(scaler_path, station_name.replace(" ", "_") + "_processed_data_scaler.pkl")
    if os.path.exists(scaler_file):
        return joblib.load(scaler_file)
    else:
        print(f"Scaler file {scaler_path} not found.")
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
