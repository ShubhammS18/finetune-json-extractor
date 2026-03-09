import json
from app.config import settings

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models prepend."""
    if '</think>' in text:
        return text.split('</think>', 1)[-1].strip()
    return text.strip()

def is_valid_json(text: str) -> bool:
    """Return True if `text` is a valid JSON string, False otherwise."""
    try:
        json.loads(strip_think_tags(text).strip())
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def field_exact_match(predicted: dict, reference: dict) -> float:
    """
    Compare predicted vs reference for each required field.

    - String values: case-insensitive, whitespace-stripped comparison.
    - List values: order-insensitive, case-insensitive set comparison.
    - Other types: direct equality.
    - Fields absent in reference (None) are skipped.

    Returns a ratio (0.0 – 1.0) of correct fields over scoreable fields.
    """
    fields = settings.required_fields
    correct = 0
    total = 0

    for field in fields:
        ref_val = reference.get(field)

        # Skip fields that have no reference value
        if ref_val is None:
            continue

        total += 1
        pred_val = predicted.get(field)

        if isinstance(ref_val, str) and isinstance(pred_val, str):
            if ref_val.lower().strip() == pred_val.lower().strip():
                correct += 1

        elif isinstance(ref_val, list) and isinstance(pred_val, list):
            if {str(x).lower() for x in ref_val} == {str(x).lower() for x in pred_val}:
                correct += 1

        else:
            # Handles: type mismatch (always False) and non-str/list types
            if ref_val == pred_val:
                correct += 1

    return correct / total if total > 0 else 0.0


def refusal_correctness(predicted_text: str, reference: dict) -> bool:
    """
    Check whether the model correctly refused to hallucinate values for
    fields that are absent (None) in the reference.

    Returns:
        True  — the model did NOT hallucinate values for any absent field.
        False — the model hallucinated at least one value, OR predicted_text
                is not valid JSON.

    Note: This function only evaluates absent-field behavior.
    It does NOT score accuracy on fields that are present in the reference;
    use field_exact_match() for that.
    """
    try:
        predicted = json.loads(strip_think_tags(predicted_text).strip())
    except (json.JSONDecodeError, TypeError):
        return False

    for field in settings.required_fields:
        if reference.get(field) is None:
            pred_val = predicted.get(field)
            # Any non-empty prediction for an absent field is a hallucination
            if pred_val not in (None, "", []):
                return False

    return True


def score_extraction(predicted_text: str, reference: dict) -> dict:
    """
    Produce a combined scoring dict for a single extraction prediction.

    Keys:
        json_valid (bool) — whether the output is parseable JSON.
        field_exact_match (float) — ratio of correctly extracted fields.
        refusal_correct (bool)  — model did not hallucinate absent fields.
    """
    predicted = {}
    valid = False

    # Single parse — result reused by both field_exact_match and the flag
    try:
        predicted = json.loads(strip_think_tags(predicted_text).strip())
        valid = True
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "json_valid": valid,
        "field_exact_match": field_exact_match(predicted, reference),
        "refusal_correct": refusal_correctness(predicted_text, reference)}