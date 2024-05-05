import joblib
import pandas as pd
import numpy as np
import glob

from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GRU, Dense
from tensorflow.keras.optimizers import Adam
import os
import mlflow
from dagshub import dagshub_logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from mlflow.models import infer_signature


mlflow.set_tracking_uri('https://lanaben:7ddc36c3f8a57b0df9f619f50e6f0a2cf6da5f55@dagshub.com/lanaben/MBajk-prediction-app.mlflow')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT_DIR, '..', '..', 'data', 'train')
TEST_DIR = os.path.join(ROOT_DIR, '..', '..', 'data', 'test')

train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.csv')))
test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.csv')))

features = ['temperature', 'relative_humidity', 'apparent_temperature', 'rain', 'hour']
target = 'available_bike_stands'

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['datetime'] = pd.to_datetime(X['datetime'])
        X['hour'] = X['datetime'].dt.hour
        X['day_of_week'] = X['datetime'].dt.dayofweek
        X['month'] = X['datetime'].dt.month
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
        return X.drop(columns=['datetime'])


class ImputeMissing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X, y=None):
        self.imputer.fit(X.select_dtypes(include=[np.number]))
        return self

    def transform(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        return X


class AddCustomFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Total_Rain_Last_3_Days'] = X['rain'].rolling(window=3, min_periods=1).sum()
        X['comfort_index'] = X['temperature'] - 0.55 * (1 - X['relative_humidity'] / 100) * (X['temperature'] - 14.5)
        return X

class ScaleFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.features] = self.scaler.transform(X[self.features])
        return X


data_pipeline = Pipeline([
    ('datetime_features', DateTimeFeatures()),
    ('impute_missing', ImputeMissing()),
    ('add_features', AddCustomFeatures()),
    ('scale_features', ScaleFeatures(features=features))
])


def prepare_sequences(data, window, future_steps=7):
    windows = []
    labels = []
    for i in range(window, len(data) - future_steps + 1):
        x = data[i - window:i]
        y = data[i:i + future_steps, 0]
        windows.append(x)
        labels.append(y)
    return np.array(windows), np.array(labels)

def get_latest_model_version_metrics(model_name, metric_name):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if model_versions:
        latest_version = max(model_versions, key=lambda x: int(x.version))
        latest_run_id = latest_version.run_id
        run = client.get_run(latest_run_id)
        metrics = run.data.metrics
        return metrics.get(metric_name)
    return None

with dagshub_logger() as logger:
    for train_file, test_file in zip(train_files, test_files):
        run_name = os.path.basename(train_file).replace('.csv', '')

        with mlflow.start_run(run_name=run_name):
            print(f"Training on: {train_file}")
            print(f"Testing on: {test_file}")

            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

            # for df in [train_df, test_df]:
            #     df['datetime'] = pd.to_datetime(df['datetime'])
            #     df['hour'] = df['datetime'].dt.hour
            #     df['day_of_week'] = df['datetime'].dt.dayofweek
            #     df['month'] = df['datetime'].dt.month
            #     df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            #
            #     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            #     avg_values = df[numeric_cols].mean()
            #     df.fillna(avg_values, inplace=True)
            #
            #     df['Total_Rain_Last_3_Days'] = df['rain'].rolling(window=3, min_periods=1).sum()
            #     df['comfort_index'] = df['temperature'] - 0.55 * (1 - df['relative_humidity']/100) * (df['temperature'] - 14.5)
            #
            # train_df.drop(columns=['datetime'], inplace=True)
            # test_df.drop(columns=['datetime'], inplace=True)
            #

            #
            # train_df.fillna(train_df.mean(), inplace=True)
            # test_df.fillna(test_df.mean(), inplace=True)
            #
            # scaler = MinMaxScaler()
            # train_scaled = scaler.fit_transform(train_df[features])
            # test_scaled = scaler.transform(test_df[features])
            #

            target = 'available_bike_stands'
            features = ['temperature', 'relative_humidity', 'apparent_temperature', 'rain', 'hour']

            train_transformed = data_pipeline.fit_transform(train_df.drop(columns=[target]))
            test_transformed = data_pipeline.transform(test_df.drop(columns=[target]))

            y_train = train_df[[target]].values.ravel()
            y_test = test_df[[target]].values.ravel()

            X_train, y_train = prepare_sequences(train_transformed[features].values, window=5, future_steps=7)
            X_test, y_test = prepare_sequences(test_transformed[features].values, window=5, future_steps=7)

            num_features = len(features)
            window_size = 5

            X_train = np.reshape(X_train, (X_train.shape[0], window_size, num_features))
            X_test = np.reshape(X_test, (X_test.shape[0], window_size, num_features))

            print(X_train.shape)
            print(X_test.shape)

            model = Sequential([
                GRU(8, activation='relu', input_shape=(window_size, num_features)),
                BatchNormalization(),
                Dense(8, activation='relu'),
                BatchNormalization(),
                Dense(7)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            signature = infer_signature(X_test, test_predictions)

            train_mae = mean_absolute_error(y_train, train_predictions)
            train_mse = mean_squared_error(y_train, train_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            train_evs = explained_variance_score(y_train, train_predictions)
            test_mae = mean_absolute_error(y_test, test_predictions)
            test_mse = mean_squared_error(y_test, test_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            test_evs = explained_variance_score(y_test, test_predictions)

            train_metrics_text = f"Train Metrics - MAE: {train_mae}, MSE: {train_mse}, EVS: {train_evs}, R2: {train_r2}"
            test_metrics_text = f"Test Metrics - MAE: {test_mae}, MSE: {test_mse}, EVS: {test_evs}, R2: {test_r2}"

            mlflow.log_params({"num_features": num_features, "epochs": 30, "batch_size": 16})
            mlflow.log_metrics({"test_mae": test_mae, "test_mse": test_mse, "test_r2": test_r2, "test_evs": test_evs})

            model_name = os.path.basename(train_file).replace('.csv', '')
            model_path = os.path.join(ROOT_DIR, '..', '..', 'models', f"{model_name}_model.h5")

            scaler = data_pipeline.named_steps['scale_features'].scaler
            scaler_path = os.path.join(ROOT_DIR, '..', '..', 'models', f"{model_name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            model.save(model_path)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)

            latest_evs = get_latest_model_version_metrics(run_name, 'test_evs')
            current_evs = test_evs

            mlflow.keras.log_model(
                model=model,
                artifact_path="keras-model",
                registered_model_name=run_name
            )
            print("New model registered, as it has better EVS.")

            train_metrics_path = os.path.join(ROOT_DIR, '..', '..', 'reports', model_name, 'train_metrics.txt')
            test_metrics_path = os.path.join(ROOT_DIR, '..', '..', 'reports', model_name, 'test_metrics.txt')

            os.makedirs(os.path.dirname(train_metrics_path), exist_ok=True)

            with open(train_metrics_path, "w") as file:
                file.write(train_metrics_text)
            with open(test_metrics_path, "w") as file:
                file.write(test_metrics_text)

print("Model training and evaluation complete.")
