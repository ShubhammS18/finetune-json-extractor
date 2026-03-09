# LLM Fine-Tuning Benchmark Report
Generated: 2026-03-09 20:31
Task: JSON extraction from job postings | Holdout set: 100 examples
Base model: Qwen2.5-7B-Instruct | Baseline: Qwen3-8B (serverless) | Platform: Fireworks AI | Method: LoRA SFT + DPO

## Results

| Metric | Base | SFT | DPO | SFT Delta | DPO Delta |
|--------|:---:|:---:|:---:|:---:|:---:|
| JSON Validity | 53.0% | 100.0% | 100.0% | +47.0% | +47.0% |
| Field Exact Match | 27.3% | 74.3% | 74.3% | +47.1% | +47.1% |
| Refusal Accuracy | 45.0% | 95.0% | 94.0% | +50.0% | +49.0% |
