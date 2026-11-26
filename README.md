# Constructive Lean 4 Kernel and Exact Zero-Float Certificates for OpenAI's Strongly Polynomial Linear Programming

**Author:** Lluis Eriksson  
**Protocol:** `SPLP-SMALE9-CERT-1.0`  
**Licenses:** MIT (code, Lean 4 proofs, certificates) / CC-BY-4.0 (manuscript & figures)

## Overview

Independent, explicitly scoped, zero-floating-point two-layer verification companion for the discrete LP trichotomy (strict complementarity, improving recession ray, minimal-support Farkas), paired-subspace reciprocal diagonal scaling & division-free Newton determinant, monotone LCP / convex QP KKT gap decomposition, and max-norm contraction / clipped-Bellman game mechanisms of OpenAI's November 24, 2026 preprint *"Strongly polynomial linear programming"* (Smale's 9th problem).

- **Layer 1 (`lean/OpenAIStronglyPolyLP.lean`):** 18 standalone constructive Lean 4.35.0 sub-lemmas (`0` `sorry`, `0` custom `axiom`, `0` `Classical.choice`, `0` warnings; `1 [propext]`, `17 [propext, Quot.sound]`).
- **Layer 2 (`src/splp_exact.py`, `src/splp_certifier.py`):** Pure-stdlib Python `int`/`Fraction` zero-float auditor across 4 modes (`lp_trichotomy_certificate_cert`, `reciprocal_scaling_and_newton_det_cert`, `monotone_lcp_and_qp_kkt_cert`, `max_norm_contraction_and_game_cert`).
- **Canonical Atlas & Red-Team Audit:** 16 canonical certificates verified (`0.049–0.485 ms`), `80/80` pytest checks (`73/73` live adversarial forgeries rejected with exact branch regexes), and full 41-file SHA-256 integrity manifest.

## Reproducibility (5-Gate Verification)

```bash
python scripts/run_all_experiments.py
python scripts/make_figures.py
python scripts/verify_all.py
```
