# ai-privacy-toolkit: Closed-Loop Privacy Extension for Data Minimization

This submission extends the original IBM `ai-privacy-toolkit` repository, specifically the minimization workflow based on Goldsteen et al. (2022), by adding a closed-loop privacy controller around the existing minimizer. The original minimization component (`GeneralizeToRepresentative`) remains the core mechanism for reducing input data granularity while preserving model utility. The new contribution adds three security features that execute in a strict sequence during runtime and jointly improve data-protection guarantees.

Added feature files:
- `apt/minimization/closed_loop_privacy.py`
- `tests/test_pipeline_order.py`
- `tests/test_privacy_floor_enforcement.py`
- `apt/minimization/__init__.py` (export wiring)

Security features and technical purpose:
1. Distillation-stage differential privacy.
`clip_distillation_signal(...)` bounds per-record soft-label sensitivity, `privatize_soft_labels(...)` injects Gaussian noise, and `account_privacy_budget(...)` tracks cumulative epsilon usage against `epsilon_max`.
2. Privacy auditing with floor enforcement.
`run_mia_attack(...)` computes leakage indicators (AUC and advantage). `compute_privacy_risk_score(...)` fuses leakage, budget pressure, and group diversity into one normalized risk signal. `governor_step(...)` compares this risk to a configurable privacy floor and can halt training if violations are severe.
3. Anti-homogeneity safeguards.
`compute_leaf_diversity_metrics(...)`, `flag_homogeneous_groups(...)`, and `rebalance_or_merge_groups(...)` detect low-diversity groups and mitigate them. The merge operation uses a true union for flagged groups (flagged-group content is combined with the largest non-flagged group), not a mirror-copy replacement.

Function dependency and invocation sequence (inside `run_closed_loop_minimization(...)`):
1. Prepare deterministic dataset and train base model.
2. Run original APT minimization (`GeneralizeToRepresentative.fit/transform`).
3. Build transformed groups and apply anti-homogeneity checks/mitigation.
4. Clip, privatize, and account privacy budget.
5. Run MIA audit and compute composite risk.
6. Let governor adapt controls (`noise_scale`, `clip_norm`) or halt.
7. Print epoch-level telemetry for reproducibility (`epoch=... auc=... eps=... risk=... halt=...`).

All added functions include multi-line docstrings and inline comments to document dependencies, control flow, and intended security roles.

## Reproducible Run Instructions

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest tests.test_pipeline_order tests.test_privacy_floor_enforcement -v
python -c "from apt.minimization.closed_loop_privacy import run_closed_loop_minimization, default_closed_loop_config; run_closed_loop_minimization(default_closed_loop_config())"
```

## References

- Goldsteen, A., Ezov, G., Shmelkin, R., Moffie, M., & Farkash, A. (2022). *Data minimization for GDPR compliance in machine learning models*. AI and Ethics, 2(3), 477-491.
- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*.
- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). *Membership Inference Attacks Against Machine Learning Models*.
