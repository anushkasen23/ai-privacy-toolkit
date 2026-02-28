"""
Module providing data minimization for ML.

This module implements a first-of-a-kind method to help reduce the amount of personal data needed to perform
predictions with a machine learning model, by removing or generalizing some of the input features. For more information
about the method see: http://export.arxiv.org/pdf/2008.04113

The main class, ``GeneralizeToRepresentative``, is a scikit-learn compatible ``Transformer``, that receives an existing
estimator and labeled training data, and learns the generalizations that can be applied to any newly collected data for
analysis by the original model. The ``fit()`` method learns the generalizations and the ``transform()`` method applies
them to new data.

It is also possible to export the generalizations as feature ranges.
"""

try:  # pragma: no cover - optional runtime dependency chain (ART, torch, etc.)
    from apt.minimization.minimizer import GeneralizeToRepresentative
except Exception:  # pragma: no cover
    GeneralizeToRepresentative = None

from apt.minimization.closed_loop_privacy import (
    account_privacy_budget,
    clip_distillation_signal,
    compute_leaf_diversity_metrics,
    compute_leakage_exposure_score,
    compute_privacy_risk_score,
    default_closed_loop_config,
    flag_homogeneous_groups,
    governor_step,
    privatize_soft_labels,
    rebalance_or_merge_groups,
    run_mia_attack,
)
from apt.minimization.closed_loop_eval import run_ablation_comparison, run_closed_loop_minimization
