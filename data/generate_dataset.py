import json, os, random, re, time
from anthropic import Anthropic
from app.config import settings

client = Anthropic(api_key=settings.anthropic_api_key)

# MODEL — Haiku
GENERATION_MODEL = 'claude-haiku-4-5-20251001'

ROLES = [
    'Machine Learning Engineer', 'Data Scientist', 'AI Engineer',
    'NLP Engineer', 'Computer Vision Engineer', 'MLOps Engineer',
    'Research Scientist', 'Data Engineer', 'AI Product Manager',
    'Deep Learning Engineer', 'LLM Engineer', 'Robotics Engineer']
COMPANIES = [
    'TechCorp', 'DataSync', 'NeuralPath', 'QuantumAI', 'DeepMind Labs',
    'CloudMatrix', 'AlgoWorks', 'InferenceIO', 'ModelHub', 'VectorSpace Inc']
CITIES = [
    'San Francisco', 'New York', 'Seattle', 'Austin', 'Boston',
    'London', 'Berlin', 'Amsterdam', 'Singapore', 'Remote']

SYSTEM_PROMPT = '''You are a recruiting copywriter.
Write realistic job postings and extract structured data.
CRITICAL: Always respond in valid JSON only. No markdown, no preamble.
For missing fields, always use null — never empty string, never omit the field.'''

SFT_TEMPLATE = '''Write a realistic job posting for a {role} at {company} in {city}.
Make it varied — different formats, abbreviations, partial info.
Some fields may be missing (e.g. no salary, no deadline).
Then extract structured data into this exact JSON schema:
{{
  "posting": "<the raw job posting text>",
  "extraction": {{
    "title": "<job title or null>",
    "company": "<company name or null>",
    "location": "<city or null>",
    "remote": <true|false|null>,
    "experience_years": <integer or null — never a string>,
    "salary_min": <integer USD annual or null — never a string>,
    "salary_max": <integer USD annual or null — never a string>,
    "skills": ["<skill1>", "<skill2>"],
    "deadline": "<YYYY-MM-DD or null>",
    "benefits": ["<benefit1>", "<benefit2>"]
  }}
}}
IMPORTANT: Use null (not empty string, not 0) for any field not mentioned.'''

def generate_sft_example(role, company, city):
    prompt = SFT_TEMPLATE.format(role=role, company=company, city=city)
    try:
        resp = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}])
        
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        data = json.loads(raw)
        if 'posting' not in data or 'extraction' not in data:
            return None
        # Validate null consistency — reject if any field uses '' instead of null
        ext = data['extraction']
        for field in ['title','company','location','experience_years',
                    'salary_min','salary_max','deadline']:
            if ext.get(field) == '' or ext.get(field) == 0:
                ext[field] = None   # standardizing to null
        return data
    except Exception as e:
        print(f'[SKIP] {e}')
        return None

def to_sft_record(example):
    return {
        'messages': [
            {'role': 'system', 'content': 'You are a JSON extraction specialist. '
             'Extract structured job data. Respond with valid JSON only.'},
            {'role': 'user', 'content': 'Extract structured data from this job posting:'
             f"\n\n{example['posting']}\n\nReturn JSON with: title, company, location, "
             'remote, experience_years, salary_min, salary_max, skills, deadline, benefits. '
             'Use null for fields not mentioned.'},
            {'role': 'assistant', 'content': json.dumps(example['extraction'])}
        ]}

def generate_dpo_pair(sft_example):
    '''Generate a DPO pair — chosen=correct, rejected=single-field-error only.
    The rejected sample must be PLAUSIBLE but wrong (not obviously broken).'''
    reject_prompt = f'''Given this job posting:
    {sft_example['posting']}
    The CORRECT extraction is:
    {json.dumps(sft_example['extraction'])}
    Generate a WRONG extraction with exactly ONE subtle error:
    - Either: hallucinate a salary that was not mentioned
    - Or: invent one skill not in the posting
    - Or: use wrong data type for one numeric field (string instead of int)
    - Or: output a value instead of null for one field that was not mentioned
    Keep all other fields identical to the correct extraction.
    Return only the wrong JSON object — no explanation.'''
    try:
        resp = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=512,
            messages=[{'role': 'user', 'content': reject_prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        rejected = json.loads(raw)
        prompt_msg = (f"Extract structured data from this job posting:\n\n{sft_example['posting']}"
                    '\n\nReturn JSON with: title, company, location, remote, experience_years,'
                    ' salary_min, salary_max, skills, deadline, benefits. Use null for missing.')
        return {
            'prompt': [
                {'role': 'system', 'content': 'You are a JSON extraction specialist. '
                'Extract structured job data. Respond with valid JSON only.'},
                {'role': 'user', 'content': prompt_msg}
            ],
            'chosen':   [{'role': 'assistant', 'content': json.dumps(sft_example['extraction'])}],
            'rejected': [{'role': 'assistant', 'content': json.dumps(rejected)}]}
        
    except Exception as e:
        print(f'[SKIP DPO] {e}')
        return None

def write_jsonl(records, path):
    os.makedirs('data', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'  Wrote {len(records)} records to {path}')

def main():
    print('='*60)
    print(' Generating SFT dataset (1,000 examples) using Claude Haiku')
    print(' Estimated Anthropic cost: ~$2.80-3.50')
    print(' Estimated time: 25-35 minutes')
    print('='*60)
    sft_examples, attempts = [], 0
    while len(sft_examples) < 1000:
        role = random.choice(ROLES)
        company = random.choice(COMPANIES)
        city = random.choice(CITIES)
        attempts += 1
        ex = generate_sft_example(role, company, city)
        if ex:
            sft_examples.append(ex)
            if len(sft_examples) % 100 == 0:
                print(f'  [{len(sft_examples)}/1000] Generated so far...')
        time.sleep(0.2)
    print(f'Generated {len(sft_examples)} valid examples in {attempts} attempts')
    sft_records = [to_sft_record(ex) for ex in sft_examples]
    random.shuffle(sft_records)
    n = len(sft_records)
    write_jsonl(sft_records[:int(n*0.8)],    'data/sft_train.jsonl')
    write_jsonl(sft_records[int(n*0.8):int(n*0.9)], 'data/sft_val.jsonl')
    write_jsonl(sft_records[int(n*0.9):],    'data/sft_holdout.jsonl')
    print('\nGenerating DPO preference pairs (200 pairs)...')
    dpo_sources = random.sample(sft_examples[:800], 200)
    dpo_pairs = []
    for i, ex in enumerate(dpo_sources):
        pair = generate_dpo_pair(ex)
        if pair:
            dpo_pairs.append(pair)
        if (i+1) % 50 == 0:
            print(f'  [{i+1}/200] DPO pairs...')
        time.sleep(0.2)
    random.shuffle(dpo_pairs)
    write_jsonl(dpo_pairs[:160], 'data/dpo_train.jsonl')
    write_jsonl(dpo_pairs[160:], 'data/dpo_val.jsonl')
    print('\nDataset generation complete. Check Anthropic console for usage.')

if __name__ == '__main__':
    main()
