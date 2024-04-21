import pandas as pd
import os
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json

def load_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data_frames = []
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

reference_data = load_data("reference_data")
current_data = load_data("processed_current")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('reports/report.html')

tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference_data, current_data=current_data)

test_results = tests.json()
parsed_results = json.loads(test_results)

print(json.dumps(parsed_results, indent=4))
