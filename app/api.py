import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.extractor import extract_json
from app.config import settings

app = FastAPI(
    title='JSON Extraction API',
    description='Fine-tuned Qwen2.5-7B for structured job posting extraction via Fireworks AI',
    version='1.0')

class ExtractRequest(BaseModel):
    posting: str
    model: str | None = None

class ExtractResponse(BaseModel):
    success: bool
    extracted: dict | None
    error: str | None
    latency_ms: int
    model_used: str | None

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'dpo_model': settings.dpo_model_id,
        'sft_model': settings.sft_model_id}

@app.post('/extract', response_model=ExtractResponse)
def extract(req: ExtractRequest):
    if not req.posting.strip():
        raise HTTPException(status_code=400, detail='posting cannot be empty')

    t0 = time.monotonic()
    result = extract_json(req.posting, model_id=req.model)
    latency_ms = int((time.monotonic() - t0) * 1000)

    return ExtractResponse(
        success=result['success'],
        extracted=result.get('extracted'),
        error=result.get('error'),
        latency_ms=latency_ms,
        model_used=result.get('model_used'))