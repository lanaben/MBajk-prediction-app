import csv
import os
from datetime import datetime


def read_last_row(csv_file):
    try:
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            last_row = None
            for last_row in reader: pass
            return last_row
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        return None


def merge_data(processed_bike_data_dir, processed_weather_data_dir, merged_data_dir):
    os.makedirs(merged_data_dir, exist_ok=True)
    bike_files = [f for f in os.listdir(processed_bike_data_dir) if f.endswith('_processed_bike_data.csv')]
    for bike_filename in bike_files:
        bike_data_file = os.path.join(processed_bike_data_dir, bike_filename)
        weather_filename = bike_filename.replace('processed_bike', 'raw_weather')
        weather_data_file = os.path.join(processed_weather_data_dir, weather_filename)

        last_bike_data = read_last_row(bike_data_file)
        last_weather_data = read_last_row(weather_data_file)

        if last_bike_data and last_weather_data:
            merged_file_path = os.path.join(merged_data_dir, bike_filename.replace('processed_bike', 'processed'))
            need_header = not os.path.exists(merged_file_path)
            with open(merged_file_path, 'a', newline='') as csvfile:
                fieldnames = ['datetime', 'bike_stands', 'available_bike_stands', 'temperature', 'relative_humidity',
                              'apparent_temperature', 'rain', 'visibility']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if need_header:
                    writer.writeheader()

                merged_row = [
                    last_bike_data[0],
                    last_bike_data[1],
                    last_bike_data[2],
                    last_weather_data[1],
                    last_weather_data[2],
                    last_weather_data[3],
                    last_weather_data[4],
                    last_weather_data[5]
                ]
                writer.writerow(dict(zip(fieldnames, merged_row)))
                print(f"Data written to {merged_file_path}: {merged_row}")


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, '..', '..', 'data')
    processed_bike_data_dir = os.path.join(DATA_DIR, "processed_bike_data")
    processed_weather_data_dir = os.path.join(DATA_DIR, "processed_weather_data")
    merged_data_dir = os.path.join(DATA_DIR, "processed")
    merge_data(processed_bike_data_dir, processed_weather_data_dir, merged_data_dir)
