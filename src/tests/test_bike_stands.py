import pytest
from src.serve.bike_stands import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint_invalid_json(client):
    response = client.post('/mbajk/predict/', json=None)
    assert response.status_code == 400

def test_predict_endpoint_missing_data_key(client):
    response = client.post('/mbajk/predict/', json={})
    assert response.status_code == 400

def test_predict_endpoint_invalid_data_length(client):
    response = client.post('/mbajk/predict/', json={"data": [1, 2, 3]})
    assert response.status_code == 400

def test_predict_endpoint_non_numeric_data(client):
    response = client.post('/mbajk/predict/', json={"data": [1, 2, 'a']})
    assert response.status_code == 400

def test_predict_endpoint_successful_prediction(client):
    response = client.post('/mbajk/predict/', json={"data": [1.0] * 186})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
