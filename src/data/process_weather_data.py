import json
import os
import csv
from datetime import datetime


def process_weather_data(raw_data_dir, processed_data_dir):
    os.makedirs(processed_data_dir, exist_ok=True)

    for filename in os.listdir(raw_data_dir):
        if filename.endswith('_weather_data.json'):
            file_path = os.path.join(raw_data_dir, filename)
            with open(file_path, 'r') as file:
                weather_data = json.load(file)

            csv_filename = filename.replace('_weather_data.json', '_weather_data.csv')
            csv_file_path = os.path.join(processed_data_dir, csv_filename)

            need_header = not os.path.exists(csv_file_path)

            with open(csv_file_path, 'a', newline='') as csvfile:
                fieldnames = ['datetime', 'temperature', 'relative_humidity', 'apparent_temperature', 'rain',
                              'visibility']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if need_header:
                    writer.writeheader()

                if 'current' in weather_data:
                    current_weather = weather_data['current']
                    record = {
                        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'temperature': current_weather.get('temperature_2m', ''),
                        'relative_humidity': current_weather.get('relative_humidity_2m', ''),
                        'apparent_temperature': current_weather.get('apparent_temperature', ''),
                        'rain': current_weather.get('rain', ''),
                        'visibility': current_weather.get('visibility', '')
                    }
                    writer.writerow(record)


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')
    raw_data_dir = os.path.join(DATA_DIR, "raw", "weather")
    processed_data_dir = os.path.join(DATA_DIR, "processed_weather_data")
    process_weather_data(raw_data_dir, processed_data_dir)
