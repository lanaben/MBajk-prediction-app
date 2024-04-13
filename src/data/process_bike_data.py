import json
import os
import csv
from datetime import datetime

def process_bike_data(raw_data_dir, processed_data_dir):
    os.makedirs(processed_data_dir, exist_ok=True)

    for filename in os.listdir(raw_data_dir):
        if filename.endswith('_raw_station_data.json'):
            file_path = os.path.join(raw_data_dir, filename)
            with open(file_path, 'r') as file:
                station_data = json.load(file)

            csv_filename = filename.replace('_raw_station_data.json', '_processed_bike_data.csv')
            csv_file_path = os.path.join(processed_data_dir, csv_filename)

            with open(csv_file_path, 'w', newline='') as csvfile:
                fieldnames = ['datetime', 'bike_stands', 'available_bike_stands']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                if isinstance(station_data, dict):
                    record = {
                        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'bike_stands': station_data.get('bike_stands'),
                        'available_bike_stands': station_data.get('available_bike_stands')
                    }
                    writer.writerow(record)

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')
    raw_data_dir = os.path.join(DATA_DIR, "raw")
    processed_data_dir = os.path.join(DATA_DIR, "processed_bike_data")
    process_bike_data(raw_data_dir, processed_data_dir)
