# Closed-Loop Privacy Governance for IBM AI Privacy Toolkit's Data Minimization

> For the original IBM AI Privacy Toolkit documentation, see the [upstream repository](https://github.com/IBM/ai-privacy-toolkit).

## Overview

This project builds upon IBM's [AI Privacy Toolkit](https://github.com/IBM/ai-privacy-toolkit), specifically the data minimization workflow by Goldsteen et al. (2022), which uses `GeneralizeToRepresentative` to reduce input data granularity while preserving model utility. The original minimizer is open-loop, it generalizes records but has no runtime privacy measurement or enforcement. The authors acknowledge this explicitly: Section 5.1 notes that feature sensitivity is not considered during generalization, Section 4.3 measures disclosure risk but provides no enforcement mechanism, and Section 7 describes the work as "a very initial implementation of data minimization for ML, leaving many areas of possible improvement."

This work extends the original minimizer with a **closed-loop privacy governor** : four security controls (`closed_loop_privacy.py`) orchestrated by a pipeline (`closed_loop_eval.py`) that wraps around the existing minimizer. The governor reads the minimizer's NCP generalization score and feeds it into its risk computation, creating a direct coupling where aggressive generalization triggers tighter privacy controls. Each epoch follows a strict sequence validated by `test_pipeline_order.py`: minimize → check diversity → apply DP → audit leakage → governor decides.

## Security Features

**1. Distillation-Stage Differential Privacy** (addresses §5.1, §2.3). The minimizer's teacher-model soft labels can leak membership information (Shokri et al., 2017). `clip_distillation_signal()` bounds each vector's L2 norm to enforce sensitivity, `privatize_soft_labels()` adds calibrated Gaussian noise (Dwork & Roth, 2014), and `account_privacy_budget()` tracks cumulative epsilon. Clipping must precede noising for the sensitivity bound to hold, this ordering is enforced by the pipeline.

**2. Anti-Homogeneity Safeguards** (addresses §3.1, §4.3). Generalization can collapse records into groups where all members share the same sensitive attribute, a classic l-diversity failure (Machanavajjhala et al., 2007). `compute_leaf_diversity_metrics()` computes normalized Shannon entropy and unique-value ratio per group, `flag_homogeneous_groups()` identifies violations, and `rebalance_or_merge_groups()` fixes them via union-merge with the largest non-flagged group.

**3. Membership Inference Auditing** (addresses §4.3, §7). The original toolkit measures disclosure risk but does not act on it. `run_mia_attack()` implements a threshold-sweeping attack computing AUC and advantage (Shokri et al., 2017). `compute_leakage_exposure_score()` fuses leakage with diversity metrics into a single score excluding epsilon cost, enabling fair baseline-vs-protected comparison.

**4. Closed-Loop Governor** (novel contribution, addresses §7). This is the architectural contribution that makes the system adaptive. `select_controls_from_risk()` maps the composite risk score to adapted DP parameters i.e. higher risk increases noise and tightens clipping. `governor_step()` enforces a configurable privacy floor by halting training immediately when risk exceeds `prs_max`, as validated by `test_privacy_floor_enforcement.py`. The minimizer's NCP score feeds directly into risk as "generalization pressure," so the governor responds to how aggressively features were collapsed.

## Ablation Results

`run_ablation_comparison()` compares baseline (no protections) vs protected (full governor stack) on identical data. On the **synthetic dataset**, MIA advantage drops 88% (0.115 → 0.014) and leakage exposure drops 56% (0.176 → 0.077). On the **UCI Adult dataset** (27,133 records from Goldsteen et al.'s evaluation), where baseline leakage is already minimal, the controls primarily improve group diversity — entropy rises 50% (0.310 → 0.465) — while leakage exposure still drops (0.155 → 0.147). The governor halts correctly when the floor is tightened (`prs_max=0.15`).

## Reproducible Run Instructions

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest tests.test_pipeline_order tests.test_privacy_floor_enforcement -v
python -c "from apt.minimization.closed_loop_eval import run_ablation_comparison; run_ablation_comparison()"
python -c "from apt.minimization.closed_loop_eval import run_ablation_comparison; run_ablation_comparison(dataset='adult')"
```

## References

- Goldsteen, A., Ezov, G., Shmelkin, R., Moffie, M., & Farkash, A. (2022). Data minimization for GDPR compliance in machine learning models. *AI and Ethics*, 2(3), 477–491. [Link](https://link.springer.com/article/10.1007/s43681-022-00163-7)
- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. [PDF](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *IEEE S&P*, 3–18. [Link](https://arxiv.org/abs/1610.05820)
- Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). l-Diversity: Privacy beyond k-anonymity. *ACM TKDD*, 1(1). [Link](https://dl.acm.org/doi/10.1145/1217299.1217302)