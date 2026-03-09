import json
from datetime import datetime

def load(p): return json.load(open(p))
def delta(a,b): d=a-b; return f'+{d*100:.1f}%' if d>=0 else f'{d*100:.1f}%'

b = load('evaluation/results/baseline_scores.json')
s = load('evaluation/results/sft_scores.json')
d = load('evaluation/results/dpo_scores.json')

report = f'''# LLM Fine-Tuning Benchmark Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Task: JSON extraction from job postings | Holdout set: {b['n_examples']} examples
Base model: Qwen2.5-7B-Instruct | Baseline: Qwen3-8B (serverless) | Platform: Fireworks AI | Method: LoRA SFT + DPO

## Results

| Metric | Base | SFT | DPO | SFT Delta | DPO Delta |
|--------|:---:|:---:|:---:|:---:|:---:|
| JSON Validity | {b['json_validity_rate']:.1%} | {s['json_validity_rate']:.1%} | {d['json_validity_rate']:.1%} | {delta(s['json_validity_rate'],b['json_validity_rate'])} | {delta(d['json_validity_rate'],b['json_validity_rate'])} |
| Field Exact Match | {b['field_match_avg']:.1%} | {s['field_match_avg']:.1%} | {d['field_match_avg']:.1%} | {delta(s['field_match_avg'],b['field_match_avg'])} | {delta(d['field_match_avg'],b['field_match_avg'])} |
| Refusal Accuracy | {b['refusal_accuracy']:.1%} | {s['refusal_accuracy']:.1%} | {d['refusal_accuracy']:.1%} | {delta(s['refusal_accuracy'],b['refusal_accuracy'])} | {delta(d['refusal_accuracy'],b['refusal_accuracy'])} |
'''

open('evaluation/results/benchmark_report.md','w').write(report)
print('Report written to evaluation/results/benchmark_report.md')
print(report)
