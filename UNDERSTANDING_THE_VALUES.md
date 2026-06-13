## Understanding the Values

This section defines every value the tools return, explains what it measures, and
states what you can and cannot validly conclude from it. All claims here are backed by
live-server measurements in [MEASURED_BEHAVIOR.md](MEASURED_BEHAVIOR.md).

### Core terms

**Eigenvalue spectrum** — the set of eigenvalues of a matrix built from model
internals (a Gram matrix of hidden-state vectors, or a weight matrix). The *pattern*
of these eigenvalues is the "spectral structure" everything here measures.

**Spacing ratio `<r>` (`r_ratio`, `mean_r_ratio`)** — the average ratio between
consecutive gaps in the sorted eigenvalue spectrum. It characterizes how the
eigenvalues are *distributed*:
- `<r> ≈ 0.5996` → GUE-like: eigenvalues "repel" each other, evenly spread (rigid spectrum).
- `<r> ≈ 0.386` → Poisson-like: eigenvalues fall independently, clumpy spacing.
- Other values → intermediate / non-standard structure.
- **What it's good for:** comparing the spectral structure of two matrices, or
  tracking how one model's spectrum changes over time. **What it is NOT:** a measure
  of reasoning quality, correctness, or hallucination. Measured live, text content
  quality does not move `<r>` in the expected direction.

**Spectral gap `lambda_2`** — the second eigenvalue (or the gap between the top two).
A large gap means one direction dominates the representation; a small gap means the
representation is spread across many directions. **Good for:** detecting rank
collapse / dominance (one feature swamping the rest). **Not:** a quality score.

**Manifold coherence `zeta` (`zeta_score`)** — a cosine-similarity-style measure of
alignment between activation structures (e.g., two layers). Higher = more aligned.
**Good for:** measuring *relative* alignment and its change. **Caveat:** plain cosine
alignment between deep-network layers is high by default; treat zeta as a relative
signal against your own baseline, not an absolute "alignment exists" claim, until
validated with a cross-input null (see Validation Status).

**GUE distance / Poisson distance (`gue_distance`, `poisson_distance`)** — how far the
measured spacing distribution sits from the ideal GUE vs ideal Poisson reference. The
*smaller* distance tells you which reference the spectrum resembles. **Good for:**
regime classification of a spectrum. Reliable on eigenvalue input.

**Spectral Health Index (`shi_score`, `spectral_health_score`, 0–100)** — a composite
score derived from the above. **Use with care:** on eigenvalue input it tracks
structural differences, but on *text* input it does **not** track content quality
(live: word salad scored 87.9 vs coherent prose 85.4). Treat SHI as a structural
descriptor, never as a correctness/quality verdict.

**Regime (`regime`, `spectral_regime`)** — a label (`gue_like`, `poisson_like`,
`intermediate`) classifying the spectrum. **Caveat:** the label is informative on
clear synthetic spectra but unreliable on real/varied input (live: coherent prose
labeled `poisson_like`). Prefer the raw distances over the label.

**Drift warning (`drift_warning`)** — a boolean flag intended to signal departure from
GUE rigidity. **Do not rely on it as a health verdict** — live testing showed it fires
`true` even on coherent inputs. It is not validated against model quality.

**Confidence (`confidence`, 0–1)** — the engine's self-reported confidence in the
measurement, driven mainly by sample count (number of spacings analyzed). Higher with
more eigenvalues/tokens. Indicates measurement stability, not result correctness.

**Intervention signal / delta / recommended_sigma** (from `compute_correction`) —
given a current `<r>` and a target (default 0.5996), the recommended adjustment
direction and magnitude to push the spectrum toward the target. **Good for:** driving
a control loop that nudges a spectrum toward GUE rigidity. Validated as a correct
controller (right sign, scales with distance). Whether reaching the target *improves a
real model* is a separate, unvalidated question.

### What you can validly use these values for

- **Compare spectral structure** of two matrices, checkpoints, or layers
  (`compare_models`, `manifold_audit` on eigenvalues/hidden states). Reliable.
- **Detect rank collapse / dominance** via `lambda_2` and the distances. Reliable.
- **Detect representational convergence/collapse between two paths** — the strongest
  validated use (see the diversity-monitoring results in the companion research).
- **Monitor change over time** in a single model's spectrum against its own baseline.
- **Drive a correction loop** toward a target `<r>` with `compute_correction`.

### What you must NOT conclude from these values

- **Not** that a model is hallucinating, incoherent, or "unhealthy" because `<r>` or
  SHI is low — text content quality does not track these values (measured live).
- **Not** that an absolute regime label or `drift_warning` is a quality verdict —
  both are unreliable on real input.
- **Not** that pushing `<r>` to 0.5996 improves model behavior — the controller works,
  but the *benefit* of reaching the target is unvalidated.
