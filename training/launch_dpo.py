import requests, os
from app.config import settings

BASE    = 'https://api.fireworks.ai/v1'
HEADERS = {'Authorization': f'Bearer {settings.fireworks_api_key}',
        'Content-Type': 'application/json'}

DPO_TRAIN_DATASET_ID = settings.dpo_train_dataset_id
DPO_VAL_DATASET_ID   = settings.dpo_val_dataset_id

if not settings.sft_model_id:
    raise ValueError('SFT_MODEL_ID not set in .env — run SFT first and copy the model ID')
if not DPO_TRAIN_DATASET_ID:
    raise ValueError('DPO_TRAIN_DATASET_ID not set in .env')

payload = {
    'displayName':  'json-extractor-dpo-v1',
    'warmStartFrom':  settings.sft_model_id,   # key: warm-starts from SFT
    'dataset':  f'accounts/{settings.fireworks_account_id}/datasets/{DPO_TRAIN_DATASET_ID}',
    'evaluationDataset':  f'accounts/{settings.fireworks_account_id}/datasets/{DPO_VAL_DATASET_ID}',
    'outputModel': f'accounts/{settings.fireworks_account_id}/models/json-extractor-dpo-v2',
    'loraRank':  8,
    'epochs':  2,           # fewer than SFT
    'learningRate':  0.00005,     # lower LR for DPO
    'earlyStop':  False,
    'wandbConfig': {
        'enabled': True,
        'apiKey':  settings.wandb_api_key,
        'project': 'json-extractor-finetune',
        'entity':  'shubhamsuradkar6-scaler',
        'runId':  'dpo-v2' }}


print('Launching DPO training job...')
print('SFT model used as starting point:', settings.sft_model_id)
print()

resp = requests.post(
    f'{BASE}/accounts/{settings.fireworks_account_id}/supervisedFineTuningJobs',
    headers=HEADERS, json=payload)

resp.raise_for_status()
job = resp.json()
job_id = job['name'].split('/')[-1]
print(f'DPO job launched: {job_id}')
print('Run: python training/monitor_job.py ' + job_id)
print('When complete, copy DPO model ID to .env as DPO_MODEL_ID')

