import requests, os
from app.config import settings

BASE    = 'https://api.fireworks.ai/v1'
HEADERS = {'Authorization': f'Bearer {settings.fireworks_api_key}',
        'Content-Type': 'application/json'}

SFT_TRAIN_DATASET_ID = settings.sft_train_dataset_id
SFT_VAL_DATASET_ID   = settings.sft_val_dataset_id

if not SFT_TRAIN_DATASET_ID or not SFT_VAL_DATASET_ID:
    raise ValueError('SFT_TRAIN_DATASET_ID and SFT_VAL_DATASET_ID must be set in .env'
                    ' — did you run create datasets on fireworks Ui first?')

payload = {
    'displayName': 'json-extractor-sft-v1',
    'baseModel':  settings.base_model,
    'dataset':  f'accounts/{settings.fireworks_account_id}/datasets/{SFT_TRAIN_DATASET_ID}',
    'evaluationDataset': f'accounts/{settings.fireworks_account_id}/datasets/{SFT_VAL_DATASET_ID}',
    'outputModel': f'accounts/{settings.fireworks_account_id}/models/json-extractor-sft-v2',
    'loraRank':  8,
    'epochs': 3,
    'learningRate': 0.0001,
    'earlyStop':  False,
    'isTurbo':  False,
    'wandbConfig':  {
        'enabled': True,
        'apiKey':  settings.wandb_api_key,
        'project': 'json-extractor-finetune',
        'entity': 'shubhamsuradkar6-scaler',   # replace with your W&B username
        'runId': 'sft-v1'}}

print('Launching SFT training job...')
print()

resp = requests.post(
    f'{BASE}/accounts/{settings.fireworks_account_id}/supervisedFineTuningJobs',
    headers=HEADERS, json=payload)
resp.raise_for_status()
job = resp.json()
job_id = job['name'].split('/')[-1]

print(f'SFT job launched successfully.')
print(f'Job ID: {job_id}')
print(f'State: {job.get("state", "PENDING")}')
print()
