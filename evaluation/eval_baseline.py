import json, os
import fireworks.client as fw
from tqdm import tqdm
from app.config import settings
from evaluation.metrics import score_extraction
import requests

fw.api_key = settings.fireworks_api_key
SYSTEM = ('You are a JSON extraction specialist. '
          'Extract structured job data. Respond with valid JSON only.')

def extract_with_model(posting, model):
    resp = requests.post('https://api.fireworks.ai/inference/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {settings.fireworks_api_key}',
            'Content-Type': 'application/json'},
        json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Extract structured data from this job posting:\n\n{posting}\n\nReturn JSON with: title, company, location, remote, experience_years, salary_min, salary_max, skills, deadline, benefits. Use null for fields not mentioned.'},
            ],
            'max_tokens': 2048,
            'temperature': 0})
    
    resp.raise_for_status()
    raw = resp.json()['choices'][0]['message']['content']
    
    if '</think>' in raw:
        return raw.split('</think>', 1)[-1].strip()
    return raw

def evaluate_model(model_id, label):
    holdout = [json.loads(l) for l in open('data/sft_holdout.jsonl', encoding='utf-8')]
    results = []
    for record in tqdm(holdout, desc=f'Evaluating {label}'):
        msgs = record['messages']
        user_msg  = next(m['content'] for m in msgs if m['role']=='user')
        reference = json.loads(next(m['content'] for m in msgs if m['role']=='assistant'))
        posting  = user_msg.split('\n\nReturn JSON')[0].replace(
            'Extract structured data from this job posting:\n\n', '')
        predicted  = extract_with_model(posting, model_id)
        results.append(score_extraction(predicted, reference))
    n = len(results)
    summary = {
        'model': label,
        'n_examples': n,
        'json_validity_rate': round(sum(r['json_valid'] for r in results)/n, 4),
        'field_match_avg': round(sum(r['field_exact_match'] for r in results)/n, 4),
        'refusal_accuracy': round(sum(r['refusal_correct'] for r in results)/n, 4)}
    
    os.makedirs('evaluation/results', exist_ok=True)
    path = f'evaluation/results/{label}_scores.json'
    json.dump(summary, open(path,'w'), indent=2)
    print(json.dumps(summary, indent=2))
    return summary

# eval_baseline.py:
if __name__ == '__main__':
    evaluate_model(settings.base_model, 'baseline')
