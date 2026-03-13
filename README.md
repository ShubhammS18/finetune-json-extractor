# finetune-json-extractor

# finetune-json-extractor

Fine-tuned Qwen2.5-7B on Fireworks AI to extract structured JSON from unstructured job postings — using LoRA SFT + DPO. Includes a FastAPI inference endpoint, custom evaluation suite, and full benchmark results.

---

## What it does

Job postings are messy. They're written in a dozen different formats, bury salary info in paragraphs, and mix requirements with marketing copy. This project fine-tunes a 7B model to reliably extract structured fields from any job posting — title, company, location, remote status, salary range, required skills, deadline, and benefits — and return clean JSON every time.

The fine-tuned model significantly outperforms a stronger general-purpose baseline (Qwen3-8B) on this specific extraction task, which is the whole point of fine-tuning.

---

## Demo

**Extracting a Stripe job posting — all fields populated:**

![Swagger UI - Stripe posting](assets/swagger_stripe.png)

**Handling a minimal posting — refusal accuracy in action:**

![Swagger UI - Berlin fin-tech posting](assets/swagger_berlin.png)

---

## Results

Evaluated on 100 holdout examples never seen during training:

| Metric | Baseline (Qwen3-8B) | SFT | DPO | SFT Δ | DPO Δ |
|--------|:---:|:---:|:---:|:---:|:---:|
| JSON Validity | 53.0% | 100.0% | 100.0% | +47.0% | +47.0% |
| Field Exact Match | 27.3% | 74.3% | 74.3% | +47.1% | +47.1% |
| Refusal Accuracy | 45.0% | 95.0% | 94.0% | +50.0% | +49.0% |

**JSON Validity** — did the model output parseable JSON? Baseline models often wrap output in markdown or reasoning blocks. Fine-tuned model hits 100%.

**Field Exact Match** — for each field present in the ground truth, did the model extract the correct value? String comparison is case-insensitive. Lists are order-insensitive set comparison.

**Refusal Accuracy** — for fields *absent* from the posting, did the model return `null` instead of hallucinating a value? This is the hardest metric to get right and the most important one for production use.

> The baseline uses Qwen3-8B (a newer, stronger model than the fine-tuned Qwen2.5-7B) because Qwen2.5-7B is not available on Fireworks serverless inference. Using a stronger baseline makes the fine-tuning results more conservative and honest — the fine-tuned smaller model still outperformed it by +47% on extraction accuracy.

---

## Training curves

**SFT training run:**

![W&B SFT training loss](assets/wandb_sft.png)

**DPO training run:**

![W&B DPO training loss](assets/wandb_dpo.png)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│                                                             │
│  Claude Haiku  ──►  1,200 synthetic job postings            │
│                     + ground truth JSON labels              │
│                                                             │
│  Split: 800 SFT train │ 100 SFT val │ 100 holdout           │
│         160 DPO train │ 40 DPO val                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING (Fireworks AI)                   │
│                                                             │
│  Qwen2.5-7B-Instruct                                        │
│       │                                                     │
│       ▼ LoRA SFT (800 examples, 3 epochs)                   │
│  json-extractor-sft-v2                                      │
│       │                                                     │
│       ▼ DPO (160 preference pairs)                          │
│  json-extractor-dpo-v2  ◄── final model                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 INFERENCE (Fireworks AI)                    │
│                                                             │
│  Base deployment (Qwen2.5-7B) + Multi-LoRA addon            │
│       │                                                     │
│       ▼                                                     │
│  FastAPI  ──►  POST /extract  ──►  structured JSON          │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Tool |
|-----------|------|
| Dataset generation | Claude Haiku (`claude-haiku-4-5`) |
| Base model | Qwen2.5-7B-Instruct |
| Training platform | Fireworks AI |
| Training method | LoRA SFT → DPO |
| Experiment tracking | Weights & Biases |
| Inference | Fireworks AI Multi-LoRA deployment |
| API | FastAPI + uvicorn |
| Evaluation | Custom Python suite |

---

## Project structure

```
finetune-json-extractor/
├── app/
│   ├── config.py              # pydantic-settings config
│   ├── extractor.py           # core inference function
│   └── api.py                 # FastAPI endpoints
├── data/
│   ├── generate_dataset.py    # synthetic data generation via Claude Haiku
│   ├── convert_dpo.py         # converts SFT format to Fireworks DPO format
│   └── *.jsonl                # generated datasets (gitignored)
├── training/
│   ├── launch_sft.py          # launches SFT job on Fireworks
│   └── launch_dpo.py          # launches DPO job on Fireworks
├── evaluation/
│   ├── metrics.py             # json_validity, field_exact_match, refusal_accuracy
│   ├── eval_baseline.py       # evaluates Qwen3-8B (no fine-tuning)
│   ├── eval_sft.py            # evaluates SFT model
│   ├── eval_dpo.py            # evaluates DPO model
│   ├── generate_report.py     # produces benchmark_report.md
│   └── results/
│       ├── baseline_scores.json
│       ├── sft_scores.json
│       ├── dpo_scores.json
│       └── benchmark_report.md
├── assets/
│   ├── swagger_stripe.png
│   ├── swagger_berlin.png
│   ├── wandb_sft.png
│   └── wandb_dpo.png
├── .env.example
├── requirements.txt
└── README.md
```

---

## API

### `GET /health`

Returns model config and server status.

```json
{
  "status": "ok",
  "dpo_model": "accounts/.../models/json-extractor-dpo-v2#accounts/.../deployments/...",
  "sft_model": "accounts/.../models/json-extractor-sft-v2#accounts/.../deployments/..."
}
```

### `POST /extract`

Takes a raw job posting, returns structured JSON.

**Request:**
```json
{
  "posting": "Senior Data Engineer at Stripe, Remote US. 6+ years experience with Python, Apache Spark, dbt, Snowflake, Airflow. Salary $180,000-$220,000. Apply before May 1 2026."
}
```

**Response:**
```json
{
  "success": true,
  "extracted": {
    "title": "Senior Data Engineer",
    "company": "Stripe",
    "location": null,
    "remote": true,
    "experience_years": 6,
    "salary_min": 180000,
    "salary_max": 220000,
    "skills": ["Python", "Apache Spark", "dbt", "Snowflake", "Airflow"],
    "deadline": "2026-05-01",
    "benefits": ["Medical coverage", "Dental coverage", "Vision coverage", "401k", "Unlimited PTO"]
  },
  "error": null,
  "latency_ms": 3199,
  "model_used": "accounts/.../models/json-extractor-dpo-v2#accounts/.../deployments/..."
}
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ShubhammS18/finetune-json-extractor
cd finetune-json-extractor
python -m venv myenv
myenv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys
```

Required keys in `.env`:
```
FIREWORKS_API_KEY=fw_...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...
FIREWORKS_ACCOUNT_ID=your-account-id
```

### 3. Deploy the model on Fireworks

The fine-tuned models are LoRA adapters — they need a base model deployment to run inference:

1. Go to `app.fireworks.ai` → Deployments → Create Deployment
2. Select **Qwen2.5-7B-Instruct** → A100 instance → Deploy
3. Once Ready, click Edit Deployment → enable **Multi-LoRA**
4. Register the LoRA addon via API:

```python
import requests

requests.post(
    'https://api.fireworks.ai/v1/accounts/YOUR_ACCOUNT/deployedModels',
    headers={'Authorization': 'Bearer YOUR_FIREWORKS_API_KEY'},
    json={
        'model': 'accounts/YOUR_ACCOUNT/models/json-extractor-dpo-v2',
        'deployment': 'accounts/YOUR_ACCOUNT/deployments/YOUR_DEPLOYMENT_ID',
        'displayName': 'dpo-eval',
    },
)
```

5. Update `.env` with the combined model + deployment ID:

```
DPO_MODEL_ID=accounts/YOUR_ACCOUNT/models/json-extractor-dpo-v2#accounts/YOUR_ACCOUNT/deployments/YOUR_DEPLOYMENT_ID
```

### 4. Run the API

```bash
uvicorn app.api:app --reload
```

Open `http://localhost:8000/docs` for the Swagger UI.

> ⚠️ Remember to delete the deployment when done — it bills per hour.

---

## Reproducing the training

> ⚠️ Running these scripts incurs real GPU costs on Fireworks AI. SFT ~$0.70, DPO ~$0.54. Do not run twice.

```bash
# Step 1 — Generate dataset (~35 min, ~$4 Anthropic credits)
python data/generate_dataset.py

# Step 2 — Upload datasets to Fireworks via UI
# Go to app.fireworks.ai → Datasets → Create Dataset
# Upload each of the 4 files: sft_train.jsonl, sft_val.jsonl, dpo_train.jsonl, dpo_val.jsonl
# Copy the dataset IDs into .env as SFT_TRAIN_DATASET_ID, SFT_VAL_DATASET_ID, etc.

# Step 3 — Launch SFT training
python training/launch_sft.py

# Step 4 — Launch DPO training (after SFT completes)
python training/launch_dpo.py

# Step 5 — Run evaluation
python evaluation/eval_baseline.py
python evaluation/eval_sft.py
python evaluation/eval_dpo.py
python evaluation/generate_report.py
```

---

## Cost breakdown

| Phase | Cost |
|-------|------|
| Dataset generation (Claude Haiku) | ~$4.00 |
| SFT training (Fireworks) | ~$0.70 |
| DPO training (Fireworks) | ~$0.54 |
| Evaluation + inference testing | ~$2.00 |
| **Total** | **~$7.24** |

---

*Built by [Shubham Suradkar](https://www.linkedin.com/in/shubhamsuradkar7) as part of a portfolio of production-grade AI engineering projects.*