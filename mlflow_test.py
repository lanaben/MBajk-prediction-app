import mlflow
mlflow.set_tracking_uri('https://dagshub.com/lanaben/MBajk-prediction-app.mlflow')
print(mlflow.get_tracking_uri())