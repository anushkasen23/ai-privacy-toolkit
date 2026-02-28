# Closed-Loop Privacy Governance for IBM AI Privacy Toolkit's Data Minimization

## Overview

This project extends the IBM's [AI Privacy Toolkit](https://github.com/IBM/ai-privacy-toolkit), in particular the data minimization workflow developed by Goldsteen et al. (2022), which makes use of `GeneralizeToRepresentative` to reduce the granularity of the input data while at the same time increasing the utility of the model. The original minimizer is an open-loop design, where there is no measurement of privacy during runtime. This work adds a **closed-loop privacy governor** with four security mechanisms to measure leakage, adapt protections, and halt training when a configurable privacy floor is violated.

## Added and Modified Files

| File | Purpose |
|------|---------|
| `apt/minimization/closed_loop_privacy.py` | Core security primitives: DP, anti-homogeneity, MIA auditing, governor |
| `apt/minimization/closed_loop_eval.py` | Dataset loaders (synthetic + UCI Adult), pipeline orchestration, ablation comparison |
| `tests/test_pipeline_order.py` | Validates pipeline steps execute in correct security-critical sequence |
| `tests/test_privacy_floor_enforcement.py` | Validates governor halts training when privacy floor is breached |
| `apt/minimization/__init__.py` | Export wiring for both new modules |
| `apt/__init__.py`, `apt/utils/datasets/datasets.py`, `apt/utils/models/__init__.py` | Wrapped optional imports (torch, Keras, XGBoost) in try/except for graceful degradation |

## Security Features

**1. Distillation-Stage Differential Privacy.** The minimizer's teacher-model soft labels can leak membership information (Shokri et al., 2017). `clip_distillation_signal()` bounds each vector's L2 norm to enforce sensitivity, `privatize_soft_labels()` adds calibrated Gaussian noise (Dwork & Roth, 2014), and `account_privacy_budget()` tracks cumulative epsilon against a configurable ceiling. Clipping must precede noising for the sensitivity bound to hold.

**2. Anti-Homogeneity Safeguards.** Generalization can collapse records into groups where all members share the same sensitive attribute, a classic l-diversity failure (Machanavajjhala et al., 2007). `compute_leaf_diversity_metrics()` computes normalized Shannon entropy and unique-value ratio per group, `flag_homogeneous_groups()` identifies violations, and `rebalance_or_merge_groups()` mitigates them via union-merge with the largest non-flagged group. These execute after the minimizer's transform since group composition is only known post-generalization.

**3. Membership Inference Auditing.** `run_mia_attack()` implements a threshold-sweeping attack computing AUC and advantage. `compute_leakage_exposure_score()` fuses leakage indicators with diversity metrics into a single normalized score excluding epsilon cost, enabling fair baseline-vs-protected comparison. The audit runs after all protections to measure their combined effectiveness.

**4. Closed-Loop Governor.** `select_controls_from_risk()` maps risk to adapted DP parameters, higher risk increases noise and tightens clipping. `governor_step()` enforces the privacy floor by halting immediately when risk exceeds `prs_max`. When the minimizer is active, its NCP score feeds directly into the risk computation as "generalization pressure," coupling the governor to the minimizer's output.

## Function Invocation Sequence

Inside `run_closed_loop_minimization()`, each epoch executes: (1) dataset preparation → (2) `GeneralizeToRepresentative.fit/transform` → (3) `compute_leaf_diversity_metrics` → `flag_homogeneous_groups` → `rebalance_or_merge_groups` → (4) `clip_distillation_signal` → `privatize_soft_labels` → `account_privacy_budget` → (5) `run_mia_attack` → `compute_privacy_risk_score` → (6) `governor_step` → (7) print epoch telemetry. This ordering is security-critical and validated by `test_pipeline_order.py`.

## Ablation Results

The `run_ablation_comparison()` function runs baseline (no protections) vs protected (full governor stack) on identical data. On the **synthetic dataset**, the protected run reduces MIA advantage by 88% (0.115 → 0.014) and leakage exposure by 56% (0.176 → 0.077), with group entropy improving from 0.709 to 0.963. On the **UCI Adult dataset** (27,133 training records), where baseline leakage is already minimal (AUC=0.501), the controls primarily improve group diversity - entropy rises 50% from 0.310 to 0.465 - while leakage exposure still drops (0.155 → 0.147). The governor correctly halts training when a tighter privacy floor (`prs_max=0.15`) is configured, as validated by `test_privacy_floor_enforcement.py`.

## Reproducible Run Instructions

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest tests.test_pipeline_order tests.test_privacy_floor_enforcement -v
python -c "from apt.minimization.closed_loop_eval import run_ablation_comparison; run_ablation_comparison()"
python -c "from apt.minimization.closed_loop_eval import run_ablation_comparison; run_ablation_comparison(dataset='adult')"
```

All runs produce deterministic output via fixed seeds. Per-epoch telemetry is printed to stdout.

## References

- Goldsteen, A., Ezov, G., Shmelkin, R., Moffie, M., & Farkash, A. (2022). Data minimization for GDPR compliance in machine learning models. *AI and Ethics*, 2(3), 477–491. [Link](https://link.springer.com/article/10.1007/s43681-022-00163-7)
- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. [PDF](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *IEEE S&P*, 3–18. [Link](https://arxiv.org/abs/1610.05820)
- Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). l-Diversity: Privacy beyond k-anonymity. *ACM TKDD*, 1(1). [Link](https://dl.acm.org/doi/10.1145/1217299.1217302)