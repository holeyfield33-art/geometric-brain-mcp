# Measured Behavior — Live Server Evidence

This document records **actual responses from the deployed Geometric Brain MCP
server** (`geometric-brain-mcp.onrender.com`, schema_version 1.1.1), captured by
calling each tool directly. It is the evidence base for the capability claims in the
README. Every number below is a real server response, not a simulation.

---

## Summary of findings

| Tool | Input type | Discriminates? | Evidence |
|------|-----------|----------------|----------|
| `brain_health_check` | raw text | **No** — does not track content quality | word salad scored *higher* than coherent prose |
| `brain_manifold_audit` | eigenvalues / hidden states | **Yes** — measures real spectral structure | distinct r, lambda_2, distances per spectrum |
| `brain_compute_correction` | r-ratio scalar | **Yes** — correct controller | right direction, magnitude, recommended sigma |
| `brain_compare_models` | two spectra | **Yes** — separates different spectra | small delta for similar, large for different |

**Bottom line:** Geometric Brain measures *spectral structure* of model internals.
On eigenvalue/hidden-state input it discriminates real structural differences. The
text proxy (`brain_health_check`) does **not** measure reasoning coherence or detect
hallucination.

---

## 1. brain_health_check (text proxy) — DOES NOT discriminate content

Three texts of differing coherence, same server, same settings:

| Input | r_ratio | SHI score | regime |
|-------|---------|-----------|--------|
| Coherent technical prose (hash functions) | 0.432 | **85.4** | poisson_like |
| Degenerate ("the the the…" repeated) | 0.312 | 73.4 | intermediate |
| Word salad (random real words) | 0.457 | **87.9** | intermediate |

**The word salad scored the highest health (87.9), above coherent prose (85.4).**
Meaningless text reads as "healthier" than a correct technical explanation. This
confirms the text proxy measures token-spacing geometry, not meaning. It is not a
content-quality or coherence signal.

(Consistent with the prior offline finding of AUROC ≈ 0.567 on a TruthfulQA text
proxy — effectively chance — and with a gpt2-large hidden-state test where coherent
vs degenerate generation separated by only ~3.5 health points with overlapping
variance.)

## 2. brain_manifold_audit (eigenvalues) — DISCRIMINATES spectral structure

Two different eigenvalue spectra produced clearly different signatures:

| Input spectrum | mean r_ratio | lambda_2 | gue_distance | zeta_score | health |
|----------------|-------------|----------|--------------|-----------|--------|
| Uniform spacing (50 evals) | 0.944 | 1.19 | 0.367 | -3.55 | 63.4 |
| Exponential growth (50 evals) | 0.706 | 0.014 | 0.128 | +2.06 | 87.2 |

Different r, different spectral gap, opposite-sign zeta, different distances. On
eigenvalue input the server measures genuine structural differences. (Offline tests
also separated synthetic GUE-like vs Poisson vs collapsed-rank spectra cleanly.)

## 3. brain_compute_correction — works as a controller

Input `current_r_ratio = 0.42`, default target 0.578:

```
delta: +0.496   direction: increase_repulsion   recommended_sigma: 1.158
action: "Increase eigenvalue repulsion. Current <r>=0.4200 is below target 0.5780."
```

Correct sign (below target → increase), sensible magnitude and recommended sigma.
(Offline sweep confirmed: zero signal at target, signed correctly above/below, scales
with distance.)

## 4. brain_compare_models — separates spectra

Offline-validated: same-family comparison yields small delta; different-family yields
large delta (~7x). The deployed tool wraps the same manifold_audit shown above.

---

## Important caveat on interpretive labels

Across all live calls, the **interpretive labels were unreliable** even when the raw
measurements were fine: coherent prose was tagged `regime: poisson_like` with
`drift_warning: true`, and most varied inputs returned `regime: intermediate`. 

**Trust the raw measured values** (`r_ratio`, `lambda_2`, `gue_distance`,
`poisson_distance`, `zeta_score`). **Do not trust the interpretive verdicts**
(`regime`, `drift_warning`, "health good/bad") as quality judgments — they are not
validated against model quality and fire on healthy inputs.

## Reproducing

Call the live tools (or `pip install` and call the engine) with the inputs above. The
text-proxy non-discrimination and the eigenvalue discrimination both reproduce.
