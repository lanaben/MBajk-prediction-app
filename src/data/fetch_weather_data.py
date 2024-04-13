import requests
import json
import os
import unicodedata
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')
raw_data_dir = os.path.join(DATA_DIR, "raw")
weather_data_dir = os.path.join(raw_data_dir, "weather")
os.makedirs(weather_data_dir, exist_ok=True)


def normalize_station_name(station_name):
    normalized_name = unicodedata.normalize('NFKD', station_name).encode('ASCII', 'ignore').decode('utf-8')
    return normalized_name.replace(" ", "_")

def fetch_weather_data(latitude, longitude):
    api_url = f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,rain,visibility'
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    # Load station data from previously saved files
    station_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_raw_station_data.json')]
    for filename in station_files:
        file_path = os.path.join(raw_data_dir, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check if data is a list and access the first element
        station = data[0] if isinstance(data, list) else data

        latitude = station['position']['lat']
        longitude = station['position']['lng']
        weather_data = fetch_weather_data(latitude, longitude)

        if weather_data:
            weather_file_path = os.path.join(weather_data_dir, filename.replace('station_data', 'weather_data'))
            with open(weather_file_path, 'w') as f:
                json.dump(weather_data, f, indent=4)
        else:
            print(f"Failed to fetch weather data for station: {normalize_station_name(station['name'])}")

if __name__ == "__main__":
    main()
