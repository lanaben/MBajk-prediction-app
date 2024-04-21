import os
import pandas as pd

def split_data_folder(input_folder, output_train_folder, output_test_folder, test_size=0.1):
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            data = pd.read_csv(file_path)

            data['datetime'] = pd.to_datetime(data['datetime'])
            data.sort_values('datetime', inplace=True)

            split_index = int((1 - test_size) * len(data))
            train = data[:split_index]
            test = data[split_index:]

            train.to_csv(os.path.join(output_train_folder, f'train_{filename}'), index=False)
            test.to_csv(os.path.join(output_test_folder, f'test_{filename}'), index=False)

input_folder = 'data/processed_current'
output_train_folder = 'data/train'
output_test_folder = 'data/test'

split_data_folder(input_folder, output_train_folder, output_test_folder)
