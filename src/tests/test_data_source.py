import requests
def test_mbajk_api():
    response = requests.get('https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b')
    assert response.status_code == 200
    print("MBajk API test ok")

def test_weather_api():
    response = requests.get('https://api.open-meteo.com/v1/forecast?latitude=46.55472&longitude=15.64556&current=temperature_2m,relative_humidity_2m')
    assert response.status_code == 200
    print("Weather API test ok")

if __name__ == "__main__":
    test_mbajk_api()
    test_weather_api()