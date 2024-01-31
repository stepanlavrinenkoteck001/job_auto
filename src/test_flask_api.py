import time
import json
import random
import requests
from pathlib import Path
from yaml import safe_load


def test_api(api_url, json_t):
    headers = {'content-type': 'application/json'}
    print('Start')
    time.sleep(5)
    r = requests.post(api_url, data=json.dumps(json_t), headers=headers)
    return r.json()


api_url = 'http://localhost:5500/smthelse'

with open('src/training_pipeline_a_config.yaml', 'r') as file:
    cf_a = safe_load(file)
parent_folder_path = Path().cwd()

job_descr_path = parent_folder_path / 'data' / cf_a['ingestion']['data_paths']['job_descr_path']
with open(job_descr_path, 'r') as file:
    data = json.load(file)

for d in data:
    id_value = random.randint(1, 100000)
    json_serialized = json.loads(json.dumps({'entity': d["description"], "id": id_value, 'sections': ''}))
    result = test_api(api_url, json_serialized)
    print("**", result)