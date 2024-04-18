import requests
import json
import os
import unicodedata

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')
raw_bike_data_dir = os.path.join(DATA_DIR, "raw", "bikes")
os.makedirs(raw_bike_data_dir, exist_ok=True)

API_KEY = '5e150537116dbc1786ce5bec6975a8603286526b'
CONTRACT = 'maribor'
API_URL = f'https://api.jcdecaux.com/vls/v1/stations?contract={CONTRACT}&apiKey={API_KEY}'

def normalize_station_name(station_name):
    normalized_name = unicodedata.normalize('NFKD', station_name).encode('ASCII', 'ignore').decode('utf-8')
    return normalized_name.replace(" ", "_")

def fetch_bike_data():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data from the API.")
        return None

def save_station_data(station):
    station_name = normalize_station_name(station['name'])
    file_path = os.path.join(raw_bike_data_dir, f"{station_name}_raw_station_data.json")
    with open(file_path, 'w') as file:
        json.dump(station, file, indent=4)

def main():
    stations = fetch_bike_data()
    if stations:
        for station in stations:
            save_station_data(station)
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()
