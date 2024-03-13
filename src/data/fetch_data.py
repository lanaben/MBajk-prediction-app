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

response = requests.get('https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b')

if response.status_code == 200:
    data = response.json()

    raw_data_dir = os.path.join(DATA_DIR, "raw")
    processed_data_dir = os.path.join(DATA_DIR, "processed")

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    for station in data:
        station_name = station['name']
        normalized_station_name = normalize_station_name(station_name)
        station_data = [station]

        prepared_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "station_name": normalized_station_name,
            "station_data": station_data
        }

        raw_data_file = os.path.join(raw_data_dir, f"{normalized_station_name}_raw_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(raw_data_file, 'w') as f:
            json.dump(prepared_data, f, indent=4)

        hourly_aggregated_data = {}

        for record in station_data:
            record_time = datetime.fromtimestamp(record['last_update'] / 1000)
            record_hour = record_time.replace(minute=0, second=0, microsecond=0)

            if record_hour not in hourly_aggregated_data:
                hourly_aggregated_data[record_hour] = []

            hourly_aggregated_data[record_hour].append(record)

        processed_data_file = os.path.join(processed_data_dir, f"{normalized_station_name}_processed_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        with open(processed_data_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'available_bikes', 'available_bike_stands'])
            writer.writeheader()

            for hour, records in hourly_aggregated_data.items():
                for record in records:
                    writer.writerow({'timestamp': record['last_update'], 'available_bikes': record['available_bikes'],
                                     'available_bike_stands': record['available_bike_stands']})

    print("Data saved successfully.")
else:
    print("Failed to fetch data from the API.")
