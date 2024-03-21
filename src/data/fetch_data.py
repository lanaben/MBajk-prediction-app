import requests
import json
import os
import csv
import unicodedata
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')

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

response = requests.get('https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b')

if response.status_code == 200:
    data = response.json()

    raw_data_dir = os.path.join(DATA_DIR, "raw")
    weather_data_dir = os.path.join(raw_data_dir, "weather")
    processed_data_dir = os.path.join(DATA_DIR, "processed")

    os.makedirs(weather_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    for station in data:
        station_name = station['name']
        normalized_station_name = normalize_station_name(station_name)
        station_data = [station]

        # Save raw station data
        raw_station_data_file = os.path.join(raw_data_dir, f"{normalized_station_name}_raw_station_data.json")
        with open(raw_station_data_file, 'w') as f:
            json.dump(station_data, f, indent=4)

        latitude = station['position']['lat']
        longitude = station['position']['lng']

        weather_data = fetch_weather_data(latitude, longitude)
        if weather_data:
            raw_weather_data_file = os.path.join(weather_data_dir, f"{normalized_station_name}_raw_weather_data.json")
            with open(raw_weather_data_file, 'w') as f:
                json.dump(weather_data, f, indent=4)
        else:
            print(f"Failed to fetch weather data for station: {station_name}")

        prepared_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "station_name": normalized_station_name,
            "station_data": station_data
        }

        processed_data_file = os.path.join(processed_data_dir, f"{normalized_station_name}_processed_data.csv")
        header = ['datetime', 'bike_stands', 'available_bike_stands', 'temperature', 'relative_humidity', 'apparent_temperature', 'rain', 'visibility']
        with open(processed_data_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)

            if os.stat(processed_data_file).st_size == 0:
                writer.writeheader()

            record_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow({
                'datetime': record_time,
                'bike_stands': station['bike_stands'],
                'available_bike_stands': station['available_bike_stands'],
                'temperature': weather_data['current']['temperature_2m'],
                'relative_humidity': weather_data['current']['relative_humidity_2m'],
                'apparent_temperature': weather_data['current']['apparent_temperature'],
                'rain': weather_data['current']['rain'],
                'visibility': weather_data['current']['visibility'],
            })

    print("Data saved successfully.")
else:
    print("Failed to fetch data from the API.")
