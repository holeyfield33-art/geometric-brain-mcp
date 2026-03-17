"""
Geometric Brain - Hidden State Hallucination Validation
=======================================================

Bridge script: HuggingFace model -> hidden states -> manifold_audit()

Tests whether eigenvalue spacing on ACTUAL hidden states can
separate hallucinated from truthful text.

Previous result (text proxy): AUROC = 0.567 (failed)
This test: hidden states from real model inference

Run in Colab:
  1. Upload spectral_engine.py to the Colab working directory
  2. Upload this file
  3. !pip install datasets scikit-learn transformers accelerate
  4. !python bridge_validation.py

Or run locally with a GPU.

TMRP: Gemini (original script) -> Claude (interface fixes + review)
"""

import json
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_engine import manifold_audit


def hidden_state_health_check(
    text: str,
    model,
    tokenizer,
    layer: int = -1,
) -> dict:
    """
    Extracts hidden states from a transformer and runs spectral audit.

    Returns: mean_r_ratio, spectral_regime, spectral_health_score, confidence
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # hidden_states is tuple of (num_layers+1) tensors
        # each shaped (batch=1, seq_len, hidden_dim)
        selected = outputs.hidden_states[layer]

        # Convert to (seq_len, hidden_dim) numpy array
        states = selected[0].float().cpu().numpy()

    # Need at least 3 tokens for meaningful spectral analysis
    if states.shape[0] < 3:
        return {
            "mean_r_ratio": None,
            "spectral_regime": "insufficient_data",
            "spectral_health_score": 0.0,
            "confidence": 0.0,
            "seq_len": states.shape[0],
        }

    # Run manifold audit from spectral engine
    result = manifold_audit(hidden_states=states.tolist())

    return {
        "mean_r_ratio": result.get("mean_r_ratio"),
        "spectral_regime": result.get("spectral_regime", "unknown"),
        "spectral_health_score": result.get("spectral_health_score", 0.0),
        "confidence": result.get("confidence", 0.0),
        "gue_distance": result.get("gue_distance"),
        "poisson_distance": result.get("poisson_distance"),
        "lambda_2": result.get("lambda_2"),
        "seq_len": states.shape[0],
        "status": result.get("status", "error"),
    }


def validate():
    print("=" * 60)
    print("GEOMETRIC BRAIN - HIDDEN STATE VALIDATION")
    print("=" * 60)

    # Model selection - try Gemma first, fall back to TinyLlama
    # TinyLlama doesn't need auth token
    model_candidates = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "google/gemma-2b",
    ]

    model = None
    tokenizer = None
    model_name = None

    for candidate in model_candidates:
        try:
            print(f"\nLoading {candidate}...")
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Use float16 if GPU available, float32 on CPU
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                model = model.to("cpu")

            model_name = candidate
            print(f"Loaded {candidate} on {model.device}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if model is None:
        print("ERROR: No model could be loaded. Exiting.")
        return

    # Load TruthfulQA
    print("\nLoading TruthfulQA dataset...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    max_samples = 100
    truthful_results = []
    hallucinated_results = []
    count = 0

    print(f"\nRunning hidden state audit on {max_samples * 2} samples...")
    print(f"Model: {model_name}")
    print()

    start_time = time.time()

    for item in ds:
        if count >= max_samples:
            break

        question = item["question"]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]

        # Find correct and incorrect answers
        correct = None
        incorrect = None
        for choice, label in zip(choices, labels):
            if label == 1 and correct is None:
                correct = choice
            elif label == 0 and incorrect is None:
                incorrect = choice
            if correct and incorrect:
                break

        if not correct or not incorrect:
            continue

        # Audit truthful
        try:
            t_result = hidden_state_health_check(f"{question} {correct}", model, tokenizer)
            if t_result["mean_r_ratio"] is not None:
                truthful_results.append(t_result)
        except Exception as e:
            print(f"  Error on truthful sample {count}: {e}")

        # Audit hallucinated
        try:
            h_result = hidden_state_health_check(f"{question} {incorrect}", model, tokenizer)
            if h_result["mean_r_ratio"] is not None:
                hallucinated_results.append(h_result)
        except Exception as e:
            print(f"  Error on hallucinated sample {count}: {e}")

        count += 1
        if count % 25 == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed
            print(
                f"  [{count}/{max_samples}] "
                f"({elapsed:.0f}s, {rate:.1f} pairs/s) "
                f"Truth <r>={np.mean([r['mean_r_ratio'] for r in truthful_results]):.4f}  "
                f"Halluc <r>={np.mean([r['mean_r_ratio'] for r in hallucinated_results]):.4f}"
            )

    elapsed = time.time() - start_time

    # Extract scores
    t_r = np.array([r["mean_r_ratio"] for r in truthful_results])
    h_r = np.array([r["mean_r_ratio"] for r in hallucinated_results])
    t_shi = np.array([r["spectral_health_score"] for r in truthful_results])
    h_shi = np.array([r["spectral_health_score"] for r in hallucinated_results])

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"\nModel: {model_name}")
    print(f"Samples: {len(t_r)} truthful, {len(h_r)} hallucinated")
    print(f"Time: {elapsed:.1f}s")

    print("\n--- Spacing Ratio <r> ---")
    print(f"  Truthful:      mean={t_r.mean():.5f}  std={t_r.std():.5f}")
    print(f"  Hallucinated:  mean={h_r.mean():.5f}  std={h_r.std():.5f}")
    print(f"  Separation:    {abs(t_r.mean() - h_r.mean()):.5f}")

    print("\n--- SHI Score ---")
    print(f"  Truthful:      mean={t_shi.mean():.2f}  std={t_shi.std():.2f}")
    print(f"  Hallucinated:  mean={h_shi.mean():.2f}  std={h_shi.std():.2f}")
    print(f"  Separation:    {abs(t_shi.mean() - h_shi.mean()):.2f}")

    # AUROC - try both metrics, both directions
    min_len = min(len(t_shi), len(h_shi))
    all_shi = np.concatenate([t_shi[:min_len], h_shi[:min_len]])
    all_r = np.concatenate([t_r[:min_len], h_r[:min_len]])
    all_labels = np.concatenate([np.ones(min_len), np.zeros(min_len)])

    auroc_shi = roc_auc_score(all_labels, all_shi)
    auroc_shi_inv = roc_auc_score(all_labels, -all_shi)
    auroc_r = roc_auc_score(all_labels, all_r)
    auroc_r_inv = roc_auc_score(all_labels, -all_r)

    best_shi = max(auroc_shi, auroc_shi_inv)
    best_r = max(auroc_r, auroc_r_inv)
    best_overall = max(best_shi, best_r)

    print("\n--- AUROC ---")
    print(f"  SHI Score:     {auroc_shi:.4f}  (inv: {auroc_shi_inv:.4f})  best: {best_shi:.4f}")
    print(f"  R-Ratio:       {auroc_r:.4f}  (inv: {auroc_r_inv:.4f})  best: {best_r:.4f}")
    print(f"  Best Overall:  {best_overall:.4f}")
    print("\n  Text proxy was: 0.5670")
    print(f"  Improvement:    {best_overall - 0.567:+.4f}")

    # Verdict
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")

    if best_overall >= 0.85:
        verdict = "GO"
        msg = (
            "AUROC >= 0.85. Hidden states work. Ship the product.\n"
            "The Geometric Brain detects hallucination from spectral health."
        )
    elif best_overall >= 0.70:
        verdict = "SIGNAL"
        msg = (
            "AUROC 0.70-0.85. Strong signal. Hidden states carry spectral\n"
            "information about truthfulness. Tune thresholds and ship."
        )
    elif best_overall >= 0.60:
        verdict = "WEAK"
        msg = (
            "AUROC 0.60-0.70. Some signal but not enough for a product.\n"
            "Try: different layers, larger model, or combined features."
        )
    else:
        verdict = "NO SIGNAL"
        msg = (
            "AUROC < 0.60. Hidden states don't separate either.\n"
            "The framework may need a fundamentally different approach\n"
            "to connecting spectral rigidity to truthfulness."
        )

    print(f"\n  >>> {verdict} <<<")
    print(f"\n  {msg}")

    # Save
    output = {
        "model": model_name,
        "truthful_count": len(t_r),
        "hallucinated_count": len(h_r),
        "truthful_r_mean": float(t_r.mean()),
        "hallucinated_r_mean": float(h_r.mean()),
        "truthful_shi_mean": float(t_shi.mean()),
        "hallucinated_shi_mean": float(h_shi.mean()),
        "separation_r": float(abs(t_r.mean() - h_r.mean())),
        "separation_shi": float(abs(t_shi.mean() - h_shi.mean())),
        "auroc_best": float(best_overall),
        "auroc_shi_best": float(best_shi),
        "auroc_r_best": float(best_r),
        "text_proxy_auroc": 0.567,
        "improvement_over_text": float(best_overall - 0.567),
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open("hidden_state_validation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to hidden_state_validation.json")

    return output


if __name__ == "__main__":
    validate()
