import pytest
from src.serve.bike_stands import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_get_station_data(client):
    station_name = 'KORO%C5%A0KA%20C.%20-%20KORO%C5%A0KI%20VETER'
    limit = 7
    response = client.get(f'/mbajk/{station_name}/{limit}')
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert 'data' in response.json
        assert isinstance(response.json['data'], list)
        assert len(response.json['data']) <= limit * 5



def test_predict(client):
    station_name = 'KORO%C5%A0KA%20C.%20-%20KORO%C5%A0KI%20VETER'
    test_data = {
        "data": [0.5] * 35
    }
    response = client.post(f'/mbajk/predict/{station_name}', json=test_data)
    assert response.status_code in [200, 400, 404]
    if response.status_code == 200:
        assert 'prediction' in response.json
        assert isinstance(response.json['prediction'], list)