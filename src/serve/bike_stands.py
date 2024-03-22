import glob
import json

import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import os

from sklearn.preprocessing import MinMaxScaler
from unidecode import unidecode

# TODO: 1.zrhitat scalerje ker nekaj ni ok (verjetno pri skranjevanju..) 2. get_station_data naj vrne v obliki, da lahko naprej passas v post prediction  3. mejbi GET trenutno stanje neke postaje

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

WINDOW_SIZE = 5
NUM_FEATURES = 5


def returnStations():
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
            print("The requested number of values exceeds the number of available entries. Returning all available entries.")
            x = len(data)

        last_values = data.tail(x)
        return last_values
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


@app.route('/mbajk/stations', methods=['GET'])
def get_stations():
    try:
        response = returnStations()
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

        response = get_last_values(normalized_name, limit)
        if response is not None and not response.empty:
            return jsonify(response.to_dict(orient='records'))
        else:
            return "No data found", 404
    except Exception as e:
        return f"An error occurred: {e}", 500


@app.route('/mbajk/predict/<station_name>', methods=['POST'])
def predict(station_name):
    if not request.is_json:
        return jsonify({"error": "Invalid JSON format"}), 400

    data = request.json

    if "data" not in data:
        return jsonify({"error": "Missing 'data' key in JSON"}), 400

    data_values = np.array(data["data"])

    scaler = load_scaler(station_name)

    if scaler is None:
        return jsonify({"error": "Scaler for the specified station could not be loaded"}), 404

    # Preverjanje, ali podatki vsebujejo pravo število vrednosti
    expected_num_features = len(scaler.get_feature_names_out())
    if len(data_values) != WINDOW_SIZE * expected_num_features:
        return jsonify({"error": f"'data' must contain exactly {WINDOW_SIZE * expected_num_features} values"}), 400

    if not all(isinstance(val, (int, float)) for val in data_values):
        return jsonify({"error": "All values in 'data' must be numbers"}), 400

    # Nalaganje modela za dano postajo
    model = load_model(station_name)
    if model is None:
        return jsonify({"error": "Model for the specified station could not be loaded"}), 404

    # Priprava podatkov za napoved
    data_array = np.array(data_values).reshape(-1, expected_num_features)
    data_normalized = scaler.transform(data_array)
    data_reshaped = data_normalized.reshape(1, WINDOW_SIZE, expected_num_features)

    # Izvajanje napovedi
    try:
        prediction_normalized = model.predict(data_reshaped)
        prediction_real = scaler.inverse_transform(prediction_normalized)
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 400

    # Vrnitev napovedi
    response = {"prediction": prediction_real.tolist()}
    return jsonify(response)




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
    print(scaler_file)
    if os.path.exists(scaler_file):
        return joblib.load(scaler_file)
    else:
        print(f"Scaler file {scaler_path} not found.")
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
