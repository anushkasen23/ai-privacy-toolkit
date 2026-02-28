"""
Closed-loop privacy security primitives for APT minimization workflows.

This module contains the reusable security controls and metrics that wrap around IBM's GeneralizeToRepresentative minimizer. 

Security controls implemented here:
    1. Distillation-stage differential privacy (clip → noise → budget)
    2. Anti-homogeneity detection and mitigation (entropy + merge)
    3. Membership inference auditing (threshold-sweep MIA)
    4. Closed-loop governor (risk → adapt controls or halt)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# data classes for structured communication between components

@dataclass
class PrivacyMetrics:
    """snapshot of all privacy metrics collected during one epoch.
    the governor reads this to decide whether to tighten controls or halt training."""

    mia_auc: float          # area under ROC curve for the MIA attack
    mia_advantage: float    # best TPR-FPR the attacker can achieve
    epsilon_spent: float    # cumulative DP budget spent so far
    leaf_entropy: float     # avg entropy of the leaf-node groups in the distillation tree
    unique_ratio: float     # avg ratio of unique sensitive values per group
    risk_score: float       # composite score combining all the above


@dataclass
class GovernorDecision:
    """what the governor tells the pipeline to do next"""

    noise_scale: float          # updated Gaussian noise scale for next epoch
    clip_norm: float            # updated L2 clip bound for next epoch
    rebalance_required: bool    # whether groups need re-merging
    halt_training: bool         # True = stop immediately, privacy floor violated


# default configuration

def default_closed_loop_config() -> Dict:
    """Return a deterministic default config for closed-loop runs."""

    return {
        # the privacy floor : if composite risk exceeds prs_max, governor halts
        "privacy_floor": {"prs_max": 0.60},

        # differential privacy parameters for the Gaussian mechanism
        "dp": {
            "epsilon_max": 3.0,
            "delta": 1e-5,
            "noise_scale": 0.80,
            "clip_norm": 1.50,
        },

        # diversity thresholds for anti-homogeneity checks
        "diversity": {"min_entropy": 0.55, "min_unique_ratio": 0.35},

        # feature flags : each control can be toggled individually
        # so the ablation can isolate their individual impact
        "features": {
            "enable_dp": True,
            "enable_anti_homogeneity": True,
            "enable_audit": True,
            "enable_governor": True,
        },

        # runtime settings for reproducibility
        "runtime": {
            "epochs": 3,
            "seed": 7,
            "sample_rate": 0.5,
            "dataset": "synthetic",         # "synthetic" or "adult"
            "require_minimizer": False,     # if True, fail instead of falling back
        },
        "debug": {"force_high_risk": False},
    }


# FEATURE 1: Distillation-stage differential privacy

def clip_distillation_signal(soft_labels: Sequence[Sequence[float]], clip_norm: float) -> List[List[float]]:
    """Clip each soft-label vector's L2 norm to bound per-record sensitivity."""

    clipped: List[List[float]] = []
    # guard against a zero clip norm which would collapse all vectors to zero
    effective_norm = max(clip_norm, 1e-12)

    for vec in soft_labels:
        norm = math.sqrt(sum(v * v for v in vec))
        # only scale down vectors that exceed the bound, leave smaller ones alone
        if norm <= effective_norm:
            clipped.append([float(v) for v in vec])
            continue
        scale = effective_norm / norm
        clipped.append([float(v) * scale for v in vec])
    return clipped


def privatize_soft_labels(
    clipped_labels: Sequence[Sequence[float]],
    noise_scale: float,
    seed: int,
) -> List[List[float]]:
    """Add Gaussian noise to already-clipped labels, then renormalise."""

    rng = random.Random(seed)
    noisy_labels: List[List[float]] = []
    sigma = max(noise_scale, 0.0)

    for vec in clipped_labels:
        # each component gets independent gaussian noise with the same sigma
        noisy = [v + rng.gauss(0.0, sigma) for v in vec]
        # clamp negatives, a probability cannot be below 0
        noisy = [max(0.0, x) for x in noisy]
        total = sum(noisy)
        # if noise wiped everything out, fall back to uniform distribution
        if total <= 0.0:
            size = len(noisy) if noisy else 1
            noisy = [1.0 / size] * size
        else:
            noisy = [x / total for x in noisy]
        noisy_labels.append(noisy)
    return noisy_labels


def account_privacy_budget(
    current_epsilon: float,
    noise_scale: float,
    sample_rate: float,
    epsilon_max: float,
    delta: float,
) -> Tuple[float, bool]:
    """Track cumulative epsilon using a simple monotonic accountant."""

    del delta     # reserved for future advanced accounting
    sigma = max(noise_scale, 1e-9)
    increment = max(0.0, sample_rate) / sigma
    new_epsilon = current_epsilon + increment
    return new_epsilon, new_epsilon <= epsilon_max


# FEATURE 2: Anti-homogeneity safeguards

def _entropy(values: Sequence[str]) -> float:
    """Normalised entropy of a list of categorical values."""

    if not values:
        return 0.0
    counts = Counter(values)
    n = len(values)
    ent = 0.0
    for count in counts.values():
        p = count / n
        ent -= p * math.log(p, 2)
    max_ent = math.log(max(len(counts), 1), 2) if counts else 1.0
    return ent / max(max_ent, 1e-12)


def compute_leaf_diversity_metrics(groups: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, float]]:
    """Compute entropy and unique-ratio for each group."""

    metrics: Dict[str, Dict[str, float]] = {}
    for group_id, values in groups.items():
        vals = list(values)
        unique_ratio = len(set(vals)) / len(vals) if vals else 0.0
        metrics[group_id] = {"entropy": _entropy(vals), "unique_ratio": unique_ratio}
    return metrics


def flag_homogeneous_groups(
    metrics: Mapping[str, Mapping[str, float]],
    min_entropy: float,
    min_unique_ratio: float,
) -> List[str]:
    """Return IDs of groups that fail either diversity threshold."""

    flagged: List[str] = []
    for group_id, item in metrics.items():
        if item.get("entropy", 0.0) < min_entropy or item.get("unique_ratio", 0.0) < min_unique_ratio:
            flagged.append(group_id)
    return flagged


def rebalance_or_merge_groups(
    groups: Mapping[str, Sequence[str]],
    flagged_groups: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, List[str]], int]:
    """Fix homogeneous groups by merging them with diverse ones."""

    rng = random.Random(seed)
    updated = {k: list(v) for k, v in groups.items()}
    interventions = 0

    flagged_set = set(flagged_groups)
    non_flagged = [k for k in updated if k not in flagged_set]
    # build a global pool in case we need the resampling fallback
    pool = [x for values in updated.values() for x in values]

    for group_id in flagged_groups:
        if group_id not in updated:
            continue
        interventions += 1
        if non_flagged:
            # pick the biggest healthy group and merge into the flagged one
            target = max(non_flagged, key=lambda gid: len(updated.get(gid, [])))
            updated[group_id] = list(updated[group_id]) + list(updated[target])
        else:
            # no healthy groups left, randomly resample from the whole dataset
            current_len = len(updated[group_id])
            if pool and current_len > 0:
                updated[group_id] = [rng.choice(pool) for _ in range(current_len)]
    return updated, interventions


def _build_groups_from_transformed(
    transformed: np.ndarray,
    sensitive: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, List[str]]:
    """Bucket transformed records into groups based on age and hours features."""

    age_index = feature_names.index("age")
    if "hours" in feature_names:
        hours_index = feature_names.index("hours")
    elif "hours-per-week" in feature_names:
        hours_index = feature_names.index("hours-per-week")
    else:
        raise ValueError("Expected 'hours' or 'hours-per-week' in feature_names for grouping")

    groups: Dict[str, List[str]] = {}
    for idx, row in enumerate(transformed):
        age_bucket = int(round(float(row[age_index]) / 10.0))
        hours_bucket = int(round(float(row[hours_index]) / 10.0))
        group_id = f"{age_bucket}:{hours_bucket}"
        groups.setdefault(group_id, []).append(str(sensitive[idx]))
    return groups


# FEATURE 3: Membership inference auditing

def run_mia_attack(member_scores: Sequence[float], non_member_scores: Sequence[float]) -> Tuple[float, float]:
    """Run a threshold-sweeping membership inference attack."""

    members = [float(s) for s in member_scores]
    non_members = [float(s) for s in non_member_scores]
    if not members or not non_members:
        return 0.5, 0.0

    # AUC via pairwise comparison
    wins = 0.0
    ties = 0.0
    for m in members:
        for n in non_members:
            if m > n:
                wins += 1.0
            elif m == n:
                ties += 1.0
    auc = (wins + 0.5 * ties) / (len(members) * len(non_members))

    # sweep all unique thresholds to find the one giving max TPR-FPR 
    grid = sorted(set(members + non_members))
    if not grid:
        return auc, 0.0
    best_adv = 0.0
    for threshold in grid:
        tpr = sum(1 for x in members if x >= threshold) / len(members)
        fpr = sum(1 for x in non_members if x >= threshold) / len(non_members)
        best_adv = max(best_adv, abs(tpr - fpr))
    return auc, best_adv


# Composite risk scoring 
def compute_privacy_risk_score(
    mia_auc: float,
    mia_advantage: float,
    epsilon_spent: float,
    leaf_entropy: float,
    unique_ratio: float,
    epsilon_max: float,
    ncp_score: Optional[float] = None,
) -> float:
    """Weighted combination of all privacy signals into one 0-1 risk score."""

    leakage_component = max(0.0, (mia_auc - 0.5) / 0.5)
    advantage_component = max(0.0, mia_advantage)
    budget_component = min(max(epsilon_spent / max(epsilon_max, 1e-12), 0.0), 1.0)
    diversity_penalty = min(max((1.0 - leaf_entropy + 1.0 - unique_ratio) / 2.0, 0.0), 1.0)

    if ncp_score is not None:
        generalization_pressure = min(max(float(ncp_score), 0.0), 1.0)
        risk = (
            0.30 * leakage_component
            + 0.20 * advantage_component
            + 0.15 * budget_component
            + 0.15 * diversity_penalty
            + 0.20 * generalization_pressure
        )
    else:
        risk = (
            0.35 * leakage_component
            + 0.25 * advantage_component
            + 0.20 * budget_component
            + 0.20 * diversity_penalty
        )
    return min(max(risk, 0.0), 1.0)


def compute_leakage_exposure_score(
    mia_auc: float,
    mia_advantage: float,
    leaf_entropy: float,
    unique_ratio: float,
    ncp_score: Optional[float] = None,
) -> float:
    """Like risk score but without epsilon budget cost."""

    leakage_component = max(0.0, (mia_auc - 0.5) / 0.5)
    advantage_component = max(0.0, mia_advantage)
    diversity_penalty = min(max((1.0 - leaf_entropy + 1.0 - unique_ratio) / 2.0, 0.0), 1.0)

    if ncp_score is not None:
        generalization_pressure = min(max(float(ncp_score), 0.0), 1.0)
        score = (
            0.40 * leakage_component
            + 0.30 * advantage_component
            + 0.15 * diversity_penalty
            + 0.15 * generalization_pressure
        )
    else:
        score = 0.45 * leakage_component + 0.35 * advantage_component + 0.20 * diversity_penalty
    return min(max(score, 0.0), 1.0)


# FEATURE 4: Closed-loop governor

def select_controls_from_risk(risk_score: float, base_noise: float, base_clip: float) -> Tuple[float, float]:
    """Translate a risk score into updated DP knobs."""

    risk = min(max(risk_score, 0.0), 1.0)
    new_noise = base_noise * (1.0 + 0.8 * risk)
    new_clip = max(0.25, base_clip * (1.0 - 0.5 * risk))
    return new_noise, new_clip


def governor_step(
    metrics: PrivacyMetrics,
    privacy_floor_max_risk: float,
    current_noise: float,
    current_clip: float,
) -> GovernorDecision:
    """The core decision function: adapt controls or halt."""

    noise, clip = select_controls_from_risk(metrics.risk_score, current_noise, current_clip)
    if metrics.risk_score > privacy_floor_max_risk:
        # floor violated --> stop everything, damage is already done
        # for this epoch, continuing would only spend more budget
        return GovernorDecision(noise_scale=noise, clip_norm=clip, rebalance_required=True, halt_training=True)
    return GovernorDecision(noise_scale=noise, clip_norm=clip, rebalance_required=False, halt_training=False)
