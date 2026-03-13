import json
import requests
from app.config import settings

SYSTEM = (
    'You are a JSON extraction specialist. '
    'Extract structured job data. Respond with valid JSON only.')

USER = (
    'Extract structured data from this job posting:\n\n{posting}\n\n'
    'Return JSON with: title, company, location, remote, experience_years, '
    'salary_min, salary_max, skills, deadline, benefits. Use null for missing fields.')

def _strip_think(text: str) -> str:
    if '</think>' in text:
        return text.split('</think>', 1)[-1].strip()
    return text.strip()

def extract_json(posting: str, model_id: str = None, temperature: float = 0.0) -> dict:
    model = model_id or settings.dpo_model_id or settings.sft_model_id or settings.base_model

    resp = requests.post(
        'https://api.fireworks.ai/inference/v1/chat/completions',
        headers={
            'Accept': 'application/json',
            'Authorization': f'Bearer {settings.fireworks_api_key}',
            'Content-Type': 'application/json'},
        json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': USER.format(posting=posting)}
            ],
            'max_tokens': 2048,
            'temperature': temperature},
        timeout=60)

    if resp.status_code != 200:
        return {
            'success': False,
            'error': f'Fireworks API error {resp.status_code}: {resp.json().get("error", {}).get("message", "unknown")}',
            'raw_output': None,
            'model_used': model}

    raw = resp.json()['choices'][0]['message']['content']
    cleaned = _strip_think(raw)

    try:
        return {
            'success': True,
            'extracted': json.loads(cleaned),
            'raw_output': raw,
            'model_used': model}
        
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': str(e),
            'raw_output': raw,
            'model_used': model}