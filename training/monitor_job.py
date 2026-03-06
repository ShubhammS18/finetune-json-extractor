import requests, time, sys
from app.config import settings

BASE    = 'https://api.fireworks.ai/v1'
HEADERS = {'Authorization': f'Bearer {settings.fireworks_api_key}'}
JOB_ID  = sys.argv[1] if len(sys.argv) > 1 else input('Enter job ID: ')

print(f'Monitoring job: {JOB_ID}')
print('Polling every 30s... (Ctrl+C to stop — does NOT cancel the job)')
print()

while True:
    resp = requests.get(
        f'{BASE}/accounts/{settings.fireworks_account_id}/supervisedFineTuningJobs/{JOB_ID}',
        headers=HEADERS)
    
    data  = resp.json()
    state = data.get('state', 'UNKNOWN')
    print(f'[{time.strftime("%H:%M:%S")}] State: {state}')
    if state == 'JOB_STATE_COMPLETED':
        model_id = data.get('outputModel', '')
        print(f'\nTraining complete!')
        print(f'Model ID: {model_id}')
        print(f'Add to .env: SFT_MODEL_ID={model_id}')
        print()
        break
    elif state == 'JOB_STATE_FAILED':
        print(f'\nJob FAILED: {data.get("status", {}).get("message", "unknown")}')
        print('Check Fireworks dashboard for error details.')
        print('Fix the issue then re-run launch_sft.py — failed jobs are NOT billed.')
        break
    time.sleep(30)
