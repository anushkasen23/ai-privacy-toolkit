"""
Closed-loop privacy controls that extend APT minimization workflows.

This module keeps the original minimization method (`GeneralizeToRepresentative`)
as the core data-minimization stage and adds three security controls:
1. Distillation-stage differential privacy over model soft labels.
2. Membership-inference-based privacy auditing with a privacy floor.
3. Anti-homogeneity mitigation plus a governor that adapts DP controls.
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class PrivacyMetrics:
    """Epoch-level privacy telemetry consumed by the governor."""

    mia_auc: float
    mia_advantage: float
    epsilon_spent: float
    leaf_entropy: float
    unique_ratio: float
    risk_score: float


@dataclass
class GovernorDecision:
    """Control decision emitted by the governor."""

    noise_scale: float
    clip_norm: float
    rebalance_required: bool
    halt_training: bool


def default_closed_loop_config() -> Dict:
    """Return a deterministic config used for tests and reproducible runs."""

    return {
        "privacy_floor": {"prs_max": 0.60},
        "dp": {
            "epsilon_max": 3.0,
            "delta": 1e-5,
            "noise_scale": 0.80,
            "clip_norm": 1.50,
        },
        "diversity": {"min_entropy": 0.55, "min_unique_ratio": 0.35},
        "features": {
            "enable_dp": True,
            "enable_anti_homogeneity": True,
            "enable_audit": True,
            "enable_governor": True,
        },
        "runtime": {"epochs": 3, "seed": 7, "sample_rate": 0.5},
        "debug": {"force_high_risk": False},
    }


def _make_minimization_dataset(
    n: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Create deterministic records with explicit train/held-out split for meaningful MIA."""

    rng = random.Random(seed)
    rows: List[List[float]] = []
    labels: List[int] = []
    sensitive: List[str] = []
    feature_names = ["age", "hours", "education_tier"]

    for _ in range(n):
        # Bucketed features create repeated combinations; stochastic labels then
        # induce imperfect separability, making MIA evaluation non-degenerate.
        age = rng.choice(list(range(20, 66, 5)))
        hours = rng.choice(list(range(20, 61, 5)))
        education_tier = rng.choice([0, 1, 2])
        sens = rng.choice(["A", "B"])

        logit = -2.0 + 0.5 * education_tier + 0.04 * (hours - 40) + 0.02 * (age - 40)
        positive_prob = 1.0 / (1.0 + math.exp(-logit))
        label = 1 if rng.random() < positive_prob else 0
        rows.append([float(age), float(hours), float(education_tier)])
        labels.append(label)
        sensitive.append(sens)

    split_idx = int(n * 0.6)
    x_train = np.array(rows[:split_idx], dtype=float)
    y_train = np.array(labels[:split_idx], dtype=int)
    x_test = np.array(rows[split_idx:], dtype=float)
    y_test = np.array(labels[split_idx:], dtype=int)
    sensitive_train = np.array(sensitive[:split_idx])
    sensitive_test = np.array(sensitive[split_idx:])

    return x_train, y_train, x_test, y_test, sensitive_train, sensitive_test, feature_names


def clip_distillation_signal(soft_labels: Sequence[Sequence[float]], clip_norm: float) -> List[List[float]]:
    """Clip each soft-label vector to enforce bounded sensitivity before noising."""

    clipped: List[List[float]] = []
    effective_norm = max(clip_norm, 1e-12)

    for vec in soft_labels:
        norm = math.sqrt(sum(v * v for v in vec))
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
    """Add Gaussian noise to clipped labels and renormalize into pseudo-probabilities."""

    rng = random.Random(seed)
    noisy_labels: List[List[float]] = []
    sigma = max(noise_scale, 0.0)

    for vec in clipped_labels:
        noisy = [v + rng.gauss(0.0, sigma) for v in vec]
        noisy = [max(0.0, x) for x in noisy]
        total = sum(noisy)
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
    """Track epsilon growth using a lightweight monotonic accountant."""

    del delta
    sigma = max(noise_scale, 1e-9)
    increment = max(0.0, sample_rate) / sigma
    new_epsilon = current_epsilon + increment
    return new_epsilon, new_epsilon <= epsilon_max


def _entropy(values: Sequence[str]) -> float:
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
    """Flag groups that violate diversity constraints."""

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
    """Mitigate homogeneous groups via union-merge or resampling fallback."""

    rng = random.Random(seed)
    updated = {k: list(v) for k, v in groups.items()}
    interventions = 0

    flagged_set = set(flagged_groups)
    non_flagged = [k for k in updated if k not in flagged_set]
    pool = [x for values in updated.values() for x in values]

    for group_id in flagged_groups:
        if group_id not in updated:
            continue
        interventions += 1
        if non_flagged:
            target = max(non_flagged, key=lambda gid: len(updated.get(gid, [])))
            updated[group_id] = list(updated[group_id]) + list(updated[target])
        else:
            current_len = len(updated[group_id])
            if pool and current_len > 0:
                updated[group_id] = [rng.choice(pool) for _ in range(current_len)]
    return updated, interventions


def _build_groups_from_transformed(
    transformed: np.ndarray,
    sensitive: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, List[str]]:
    """Create group ids from transformed minimization features and attach sensitive values."""

    age_index = feature_names.index("age")
    hours_index = feature_names.index("hours")
    groups: Dict[str, List[str]] = {}

    for idx, row in enumerate(transformed):
        age_bucket = int(round(float(row[age_index]) / 10.0))
        hours_bucket = int(round(float(row[hours_index]) / 10.0))
        group_id = f"{age_bucket}:{hours_bucket}"
        groups.setdefault(group_id, []).append(str(sensitive[idx]))
    return groups


def run_mia_attack(member_scores: Sequence[float], non_member_scores: Sequence[float]) -> Tuple[float, float]:
    """Compute AUC-like score and attack advantage via threshold sweeping."""

    members = [float(s) for s in member_scores]
    non_members = [float(s) for s in non_member_scores]
    if not members or not non_members:
        return 0.5, 0.0

    wins = 0.0
    ties = 0.0
    for m in members:
        for n in non_members:
            if m > n:
                wins += 1.0
            elif m == n:
                ties += 1.0
    auc = (wins + 0.5 * ties) / (len(members) * len(non_members))

    grid = sorted(set(members + non_members))
    if not grid:
        return auc, 0.0
    best_adv = 0.0
    for threshold in grid:
        tpr = sum(1 for x in members if x >= threshold) / len(members)
        fpr = sum(1 for x in non_members if x >= threshold) / len(non_members)
        best_adv = max(best_adv, abs(tpr - fpr))
    return auc, best_adv


def compute_privacy_risk_score(
    mia_auc: float,
    mia_advantage: float,
    epsilon_spent: float,
    leaf_entropy: float,
    unique_ratio: float,
    epsilon_max: float,
) -> float:
    """Combine leakage, budget pressure, and diversity into one normalized score."""

    leakage_component = max(0.0, (mia_auc - 0.5) / 0.5)
    advantage_component = max(0.0, mia_advantage)
    budget_component = min(max(epsilon_spent / max(epsilon_max, 1e-12), 0.0), 1.0)
    diversity_penalty = min(max((1.0 - leaf_entropy + 1.0 - unique_ratio) / 2.0, 0.0), 1.0)
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
) -> float:
    """Compute leakage-focused risk, excluding epsilon-budget accounting cost."""

    leakage_component = max(0.0, (mia_auc - 0.5) / 0.5)
    advantage_component = max(0.0, mia_advantage)
    diversity_penalty = min(max((1.0 - leaf_entropy + 1.0 - unique_ratio) / 2.0, 0.0), 1.0)
    score = 0.45 * leakage_component + 0.35 * advantage_component + 0.20 * diversity_penalty
    return min(max(score, 0.0), 1.0)


def select_controls_from_risk(risk_score: float, base_noise: float, base_clip: float) -> Tuple[float, float]:
    """Map risk score to new DP controls."""

    risk = min(max(risk_score, 0.0), 1.0)
    new_noise = base_noise * (1.0 + 0.8 * risk)
    new_clip = max(0.25, base_clip * (1.0 - 0.5 * risk))
    return new_noise, new_clip


def governor_step(metrics, privacy_floor_max_risk, current_noise, current_clip):
    noise, clip = select_controls_from_risk(metrics.risk_score, current_noise, current_clip)
    if metrics.risk_score > privacy_floor_max_risk:
        # Halt immediately on any floor violation - controls cannot undo
        # leakage already observed this epoch.
        return GovernorDecision(
            noise_scale=noise, clip_norm=clip, 
            rebalance_required=True, halt_training=True
        )
    return GovernorDecision(
        noise_scale=noise, clip_norm=clip, 
        rebalance_required=False, halt_training=False
    )


def run_closed_loop_minimization(config: Dict | None = None) -> Dict:
    """
    Execute minimization with closed-loop privacy controls.

    Invocation sequence:
    1. Create deterministic dataset and train base model.
    2. Run minimization (`GeneralizeToRepresentative.fit/transform`) when available.
    3. Build transformed groups and apply anti-homogeneity mitigation.
    4. Clip + privatize soft labels and account privacy budget.
    5. Run MIA audit, compute risk, and let governor adjust/halt.
    6. Print epoch summary for reproducibility.
    """

    cfg = copy.deepcopy(config) if config is not None else default_closed_loop_config()
    seed = int(cfg["runtime"]["seed"])
    epochs = int(cfg["runtime"]["epochs"])
    sample_rate = float(cfg["runtime"]["sample_rate"])

    noise_scale = float(cfg["dp"]["noise_scale"])
    clip_norm = float(cfg["dp"]["clip_norm"])
    epsilon_max = float(cfg["dp"]["epsilon_max"])
    delta = float(cfg["dp"]["delta"])
    prs_max = float(cfg["privacy_floor"]["prs_max"])

    (
        x_train,
        y_train,
        x_test,
        _y_test,
        sensitive_train,
        _sensitive_test,
        feature_names,
    ) = _make_minimization_dataset(n=240, seed=seed)
    base_model = DecisionTreeClassifier(random_state=seed, max_depth=5)
    base_model.fit(x_train, y_train)

    epsilon_spent = 0.0
    history: List[Dict] = []
    full_trace: List[List[str]] = []
    halted = False

    for epoch in range(1, epochs + 1):
        trace: List[str] = []
        trace.append("prepare_dataset_and_model")

        trace.append("fit_and_apply_minimizer")
        ncp_transform_score = None
        try:
            # Use the original APT minimizer when ART-backed dependencies are available.
            from apt.minimization.minimizer import GeneralizeToRepresentative  # type: ignore

            minimizer = GeneralizeToRepresentative(base_model, target_accuracy=0.98)
            teacher_predictions = base_model.predict(x_train)
            minimizer.fit(x_train, teacher_predictions, features_names=feature_names)
            transformed_train = minimizer.transform(x_train, features_names=feature_names)
            transformed_test = minimizer.transform(x_test, features_names=feature_names)
            transformed_train_np = np.asarray(transformed_train, dtype=float)
            transformed_test_np = np.asarray(transformed_test, dtype=float)
            ncp_transform_score = minimizer.ncp.transform_score
        except Exception:
            # Fallback keeps the security-controller path executable in lightweight environments.
            transformed_train_np = x_train
            transformed_test_np = x_test

        trace.append("diversity_flag_and_mitigate")
        groups = _build_groups_from_transformed(transformed_train_np, sensitive_train, feature_names)
        interventions = 0
        if cfg["features"].get("enable_anti_homogeneity", True):
            diversity_metrics = compute_leaf_diversity_metrics(groups)
            flagged = flag_homogeneous_groups(
                diversity_metrics,
                min_entropy=float(cfg["diversity"]["min_entropy"]),
                min_unique_ratio=float(cfg["diversity"]["min_unique_ratio"]),
            )
            groups, interventions = rebalance_or_merge_groups(groups, flagged, seed + epoch)
        diversity_metrics = compute_leaf_diversity_metrics(groups)

        trace.append("clip_privatize_and_account")
        soft_labels = base_model.predict_proba(transformed_train_np).tolist()
        if cfg["features"].get("enable_dp", True):
            clipped = clip_distillation_signal(soft_labels, clip_norm)
            private_labels = privatize_soft_labels(clipped, noise_scale, seed + epoch)
            epsilon_spent, _ = account_privacy_budget(epsilon_spent, noise_scale, sample_rate, epsilon_max, delta)
        else:
            private_labels = [list(v) for v in soft_labels]

        private_hard_labels = [int(vec[1] >= 0.5) for vec in private_labels]
        surrogate = DecisionTreeClassifier(random_state=seed + epoch, max_depth=4)
        surrogate.fit(transformed_train_np, private_hard_labels)

        trace.append("run_mia_and_compute_risk")
        if cfg["features"].get("enable_audit", True):
            member_scores = [max(v) for v in surrogate.predict_proba(transformed_train_np).tolist()]
            non_member_scores = [max(v) for v in surrogate.predict_proba(transformed_test_np).tolist()]
            mia_auc, mia_advantage = run_mia_attack(member_scores, non_member_scores)
        else:
            mia_auc, mia_advantage = 0.5, 0.0

        mean_entropy = sum(m["entropy"] for m in diversity_metrics.values()) / max(len(diversity_metrics), 1)
        mean_unique = sum(m["unique_ratio"] for m in diversity_metrics.values()) / max(len(diversity_metrics), 1)
        risk_score = compute_privacy_risk_score(
            mia_auc=mia_auc,
            mia_advantage=mia_advantage,
            epsilon_spent=epsilon_spent,
            leaf_entropy=mean_entropy,
            unique_ratio=mean_unique,
            epsilon_max=epsilon_max,
        )
        if cfg.get("debug", {}).get("force_high_risk"):
            risk_score = 1.0

        trace.append("governor_adjust_or_halt")
        metrics = PrivacyMetrics(
            mia_auc=mia_auc,
            mia_advantage=mia_advantage,
            epsilon_spent=epsilon_spent,
            leaf_entropy=mean_entropy,
            unique_ratio=mean_unique,
            risk_score=risk_score,
        )
        if cfg["features"].get("enable_governor", True):
            decision = governor_step(metrics, prs_max, noise_scale, clip_norm)
            noise_scale = decision.noise_scale
            clip_norm = decision.clip_norm
        else:
            decision = GovernorDecision(
                noise_scale=noise_scale, clip_norm=clip_norm, rebalance_required=False, halt_training=False
            )

        trace.append("print_epoch_summary")
        print(
            f"epoch={epoch} auc={mia_auc:.3f} adv={mia_advantage:.3f} "
            f"eps={epsilon_spent:.3f} ent={mean_entropy:.3f} uniq={mean_unique:.3f} "
            f"risk={risk_score:.3f} interventions={interventions} halt={decision.halt_training}"
        )

        history.append(
            {
                "epoch": epoch,
                "metrics": metrics,
                "decision": decision,
                "interventions": interventions,
                "ncp_transform_score": ncp_transform_score,
            }
        )
        full_trace.append(trace)

        if decision.halt_training:
            halted = True
            break

    return {"history": history, "trace": full_trace, "halted": halted, "epochs_ran": len(history)}


def run_ablation_comparison(seed: int = 7) -> None:
    """Run baseline vs protected modes and print a compact comparison summary."""

    print("\n" + "=" * 60)
    print("ABLATION: BASELINE (no privacy protections, 3 epochs)")
    print("=" * 60)
    baseline_cfg = default_closed_loop_config()
    baseline_cfg["runtime"]["epochs"] = 3
    baseline_cfg["runtime"]["seed"] = seed
    baseline_cfg["privacy_floor"]["prs_max"] = 1.0
    baseline_cfg["features"] = {
        "enable_dp": False,
        "enable_anti_homogeneity": False,
        "enable_audit": True,
        "enable_governor": False,
    }
    baseline_result = run_closed_loop_minimization(baseline_cfg)
    baseline_metrics = max([item["metrics"] for item in baseline_result["history"]], key=lambda metric: metric.risk_score)

    print("\n" + "=" * 60)
    print("PROTECTED: Full closed-loop governor stack (3 epochs)")
    print("=" * 60)
    protected_cfg = default_closed_loop_config()
    protected_cfg["runtime"]["epochs"] = 3
    protected_cfg["runtime"]["seed"] = seed
    protected_result = run_closed_loop_minimization(protected_cfg)
    protected_metrics = max([item["metrics"] for item in protected_result["history"]], key=lambda metric: metric.risk_score)

    baseline_leakage = compute_leakage_exposure_score(
        baseline_metrics.mia_auc, baseline_metrics.mia_advantage, baseline_metrics.leaf_entropy, baseline_metrics.unique_ratio
    )
    protected_leakage = compute_leakage_exposure_score(
        protected_metrics.mia_auc, protected_metrics.mia_advantage, protected_metrics.leaf_entropy, protected_metrics.unique_ratio
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (worst-case epoch per run)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Protected':>12} {'Improvement':>12}")
    print("-" * 60)
    print(
        f"{'MIA AUC':<20} {baseline_metrics.mia_auc:>12.3f} "
        f"{protected_metrics.mia_auc:>12.3f} "
        f"{'down better' if protected_metrics.mia_auc < baseline_metrics.mia_auc else '-':>12}"
    )
    print(
        f"{'MIA Advantage':<20} {baseline_metrics.mia_advantage:>12.3f} "
        f"{protected_metrics.mia_advantage:>12.3f} "
        f"{'down better' if protected_metrics.mia_advantage < baseline_metrics.mia_advantage else '-':>12}"
    )
    print(
        f"{'Leakage Score':<20} {baseline_leakage:>12.3f} "
        f"{protected_leakage:>12.3f} "
        f"{'down better' if protected_leakage < baseline_leakage else '-':>12}"
    )
    print(
        f"{'Leaf Entropy':<20} {baseline_metrics.leaf_entropy:>12.3f} "
        f"{protected_metrics.leaf_entropy:>12.3f} "
        f"{'up better' if protected_metrics.leaf_entropy > baseline_metrics.leaf_entropy else '-':>12}"
    )
    print("=" * 60)

    print("\nPer-epoch counterfactual deltas (protected - baseline):")
    print(f"{'Epoch':<8} {'dAUC':>10} {'dAdv':>10} {'dLeak':>10}")
    for epoch_idx, (b_item, p_item) in enumerate(zip(baseline_result["history"], protected_result["history"]), start=1):
        b_metrics = b_item["metrics"]
        p_metrics = p_item["metrics"]
        b_leak = compute_leakage_exposure_score(
            b_metrics.mia_auc, b_metrics.mia_advantage, b_metrics.leaf_entropy, b_metrics.unique_ratio
        )
        p_leak = compute_leakage_exposure_score(
            p_metrics.mia_auc, p_metrics.mia_advantage, p_metrics.leaf_entropy, p_metrics.unique_ratio
        )
        print(
            f"{epoch_idx:<8} "
            f"{(p_metrics.mia_auc - b_metrics.mia_auc):>10.3f} "
            f"{(p_metrics.mia_advantage - b_metrics.mia_advantage):>10.3f} "
            f"{(p_leak - b_leak):>10.3f}"
        )
