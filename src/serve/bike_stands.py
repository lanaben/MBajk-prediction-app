import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from keras.models import load_model

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'models')

model_data_dir = os.path.join(DATA_DIR, "bike_stands_model3.h5")
scaler_dir = os.path.join(DATA_DIR, "bike_stands_scaler.joblib")

model = load_model(model_data_dir)
scaler = joblib.load(scaler_dir)


@app.route('/mbajk/predict/', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON format"}), 400

    data = request.json

    if "data" not in data:
        return jsonify({"error": "Missing 'data' key in JSON"}), 400

    data_values = data["data"]

    if len(data["data"]) != 186:
        return jsonify({"error": "'data' must contain exactly 186 values"}), 400

    if not all(isinstance(val, (int, float)) for val in data_values):
        return jsonify({"error": "All values in 'data' must be numbers"}), 400

    data_array = np.array(data_values).reshape(-1, 1)

    try:
        data_normalized = scaler.transform(data_array)
    except Exception as e:
        return jsonify({"error": "Data normalization failed: " + str(e)}), 400

    data_reshaped = data_normalized.reshape(1, 1, 186)

    try:
        prediction = model.predict(data_reshaped)
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 400

    try:
        prediction_denormalized = scaler.inverse_transform(prediction.reshape(-1, 1))
    except Exception as e:
        return jsonify({"error": "Denormalization failed: " + str(e)}), 400

    response = {"prediction": float(prediction_denormalized[0][0])}

    prediction_value = response["prediction"]
    print("Predicted value:", prediction_value)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
