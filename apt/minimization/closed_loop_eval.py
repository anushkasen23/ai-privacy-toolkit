"""Closed-loop minimization evaluation and orchestration workflows.

This module wires together the security primitives from closed_loop_privacy.py with 
dataset loading, the original APT minimizer, and evaluation reporting."""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# all security controls come from the primitive module
from apt.minimization.closed_loop_privacy import (
    GovernorDecision,
    PrivacyMetrics,
    _build_groups_from_transformed,
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

# dataset loaders

def _make_minimization_dataset(
    n: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generate a small synthetic dataset for fast, reproducible testing."""

    rng = random.Random(seed)
    rows: List[List[float]] = []
    labels: List[int] = []
    sensitive: List[str] = []
    feature_names = ["age", "hours", "education_tier"]

    for _ in range(n):
        age = rng.choice(list(range(20, 66, 5)))
        hours = rng.choice(list(range(20, 61, 5)))
        education_tier = rng.choice([0, 1, 2])
        sens = rng.choice(["A", "B"])

        # simple logistic model: higher education and longer hours -> more likely positive, with some noise
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


def _load_adult_dataset(
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load the UCI Adult dataset used in Goldsteen et al. (2022) evaluation.
    First run downloads the data from OpenML; subsequent runs use the cache.
    """

    from sklearn.datasets import fetch_openml

    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    data = adult.frame.dropna()

    feature_cols = ["age", "hours-per-week", "education-num"]
    sensitive_col = "race"

    x_all = data[feature_cols].to_numpy(dtype=float)
    y_all = (data["class"].astype(str).str.strip().isin([">50K", ">50K."])).astype(int).to_numpy()
    sensitive_all = data[sensitive_col].astype(str).to_numpy()

    # deterministic shuffle and split so results are reproducible
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(x_all))
    split_idx = int(len(x_all) * 0.6)
    # cap test set at 1000 records to keep MIA runtime reasonable
    test_end = min(len(x_all), split_idx + 1000)

    x_train = x_all[indices[:split_idx]]
    y_train = y_all[indices[:split_idx]]
    x_test = x_all[indices[split_idx:test_end]]
    y_test = y_all[indices[split_idx:test_end]]
    sensitive_train = sensitive_all[indices[:split_idx]]
    sensitive_test = sensitive_all[indices[split_idx:test_end]]
    feature_names = ["age", "hours-per-week", "education-num"]

    return x_train, y_train, x_test, y_test, sensitive_train, sensitive_test, feature_names

# main pipeline orchestrator

def run_closed_loop_minimization(config: Dict | None = None) -> Dict:
    """Execute the full closed-loop minimization pipeline."""

    cfg = copy.deepcopy(config) if config is not None else default_closed_loop_config()
    seed = int(cfg["runtime"]["seed"])
    epochs = int(cfg["runtime"]["epochs"])
    sample_rate = float(cfg["runtime"]["sample_rate"])
    dataset_choice = str(cfg["runtime"].get("dataset", "synthetic")).lower()
    require_minimizer = bool(cfg["runtime"].get("require_minimizer", False))

    noise_scale = float(cfg["dp"]["noise_scale"])
    clip_norm = float(cfg["dp"]["clip_norm"])
    epsilon_max = float(cfg["dp"]["epsilon_max"])
    delta = float(cfg["dp"]["delta"])
    prs_max = float(cfg["privacy_floor"]["prs_max"])

    # Step 1: Load dataset
    if dataset_choice == "adult":
        (
            x_train,
            y_train,
            x_test,
            _y_test,
            sensitive_train,
            _sensitive_test,
            feature_names,
        ) = _load_adult_dataset(seed)
        print(f"[dataset] UCI Adult - {len(x_train)} train, {len(x_test)} test records")
    else:
        (
            x_train,
            y_train,
            x_test,
            _y_test,
            sensitive_train,
            _sensitive_test,
            feature_names,
        ) = _make_minimization_dataset(n=240, seed=seed)
        print(f"[dataset] Synthetic - {len(x_train)} train, {len(x_test)} test records")

    # train the base model that the minimizer will wrap around 
    base_model = DecisionTreeClassifier(random_state=seed, max_depth=5)
    base_model.fit(x_train, y_train)

    epsilon_spent = 0.0
    history: List[Dict] = []
    full_trace: List[List[str]] = []
    halted = False

    for epoch in range(1, epochs + 1):
        # the trace records which steps ran this epoch
        trace: List[str] = []
        trace.append("prepare_dataset_and_model")

        # Step 2: run the original APT minimizer
        trace.append("fit_and_apply_minimizer")
        ncp_transform_score = None
        try:
            from apt.minimization.minimizer import GeneralizeToRepresentative  # type: ignore

            minimizer = GeneralizeToRepresentative(base_model, target_accuracy=0.98)
            teacher_predictions = base_model.predict(x_train)
            minimizer.fit(x_train, teacher_predictions, features_names=feature_names)
            transformed_train = minimizer.transform(x_train, features_names=feature_names)
            transformed_test = minimizer.transform(x_test, features_names=feature_names)
            transformed_train_np = np.asarray(transformed_train, dtype=float)
            transformed_test_np = np.asarray(transformed_test, dtype=float)
            # NCP score measures how aggresively the minimizer generalised
            # we feed this into the risk score to couple the governor to the minimizer
            ncp_transform_score = minimizer.ncp.transform_score
        except Exception as exc:
            if require_minimizer:
                raise RuntimeError(
                    "GeneralizeToRepresentative could not be used, but runtime.require_minimizer=True"
                ) from exc
            # fallback: run security controls on raw features
            transformed_train_np = x_train
            transformed_test_np = x_test

        # Step 3: Anti-homogeneity checks
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
        # recompute metrics after mitigation so downstream scoring sees the fixed state
        diversity_metrics = compute_leaf_diversity_metrics(groups)

        # Step 4: Differential privacy on soft labels
        trace.append("clip_privatize_and_account")
        soft_labels = base_model.predict_proba(transformed_train_np).tolist()
        if cfg["features"].get("enable_dp", True):
            clipped = clip_distillation_signal(soft_labels, clip_norm)
            private_labels = privatize_soft_labels(clipped, noise_scale, seed + epoch)
            epsilon_spent, _ = account_privacy_budget(epsilon_spent, noise_scale, sample_rate, epsilon_max, delta)
        else:
            private_labels = [list(v) for v in soft_labels]

        # Train a surrogate on the (possibly noisy) labels, this is what the MIA attack will probe to detect membership leakage
        private_hard_labels = [int(vec[1] >= 0.5) for vec in private_labels]
        surrogate = DecisionTreeClassifier(random_state=seed + epoch, max_depth=4)
        surrogate.fit(transformed_train_np, private_hard_labels)

        # Step 5: MIA audit and risk computation
        trace.append("run_mia_and_compute_risk")
        if cfg["features"].get("enable_audit", True):
            # Member scores = surrogate confidence on training data (should be higher)
            # Non-member scores = surrogate confidence on held-out data
            member_scores = [max(v) for v in surrogate.predict_proba(transformed_train_np).tolist()]
            non_member_scores = [max(v) for v in surrogate.predict_proba(transformed_test_np).tolist()]
            mia_auc, mia_advantage = run_mia_attack(member_scores, non_member_scores)
        else:
            mia_auc, mia_advantage = 0.5, 0.0

        # aggregate diversity metrics across all groups for the risk formula
        mean_entropy = sum(m["entropy"] for m in diversity_metrics.values()) / max(len(diversity_metrics), 1)
        mean_unique = sum(m["unique_ratio"] for m in diversity_metrics.values()) / max(len(diversity_metrics), 1)

        # Composite risk score, feeds into governor decision
        risk_score = compute_privacy_risk_score(
            mia_auc=mia_auc,
            mia_advantage=mia_advantage,
            epsilon_spent=epsilon_spent,
            leaf_entropy=mean_entropy,
            unique_ratio=mean_unique,
            epsilon_max=epsilon_max,
            ncp_score=ncp_transform_score,
        )
        if cfg.get("debug", {}).get("force_high_risk"):
            risk_score = 1.0

        # Step 6: Governor decision
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
            # governor's updated controls carry forward to the next epoch
            noise_scale = decision.noise_scale
            clip_norm = decision.clip_norm
        else:
            decision = GovernorDecision(
                noise_scale=noise_scale,
                clip_norm=clip_norm,
                rebalance_required=False,
                halt_training=False,
            )

        # Step 7: print epoch summary
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

# Ablation comparison

def _leakage_from_item(item: Dict) -> float:
    """Helper to compute leakage exposure score from a history entry."""

    m = item["metrics"]
    return compute_leakage_exposure_score(
        m.mia_auc, m.mia_advantage, m.leaf_entropy, m.unique_ratio,
        item.get("ncp_transform_score"),
    )

def run_ablation_comparison(seed: int = 7, dataset: str = "synthetic") -> None:
    """Run baseline vs protected modes and print a compact comparison summary."""

    # Baseline: minimizer only, no security controls
    print("\n" + "=" * 60)
    print("ABLATION: BASELINE (no privacy protections, 3 epochs)")
    print("=" * 60)
    baseline_cfg = default_closed_loop_config()
    baseline_cfg["runtime"]["epochs"] = 3
    baseline_cfg["runtime"]["seed"] = seed
    baseline_cfg["runtime"]["dataset"] = dataset
    baseline_cfg["privacy_floor"]["prs_max"] = 1.0  # effectively disable floor
    baseline_cfg["features"] = {
        "enable_dp": False,
        "enable_anti_homogeneity": False,
        "enable_audit": True,           # still measure leakage, just don't act on it
        "enable_governor": False,
    }
    baseline_result = run_closed_loop_minimization(baseline_cfg)
    baseline_worst = max(baseline_result["history"], key=lambda item: item["metrics"].risk_score)
    baseline_metrics = baseline_worst["metrics"]

    # Protected: all controls active
    print("\n" + "=" * 60)
    print("PROTECTED: Full closed-loop governor stack (3 epochs)")
    print("=" * 60)
    protected_cfg = default_closed_loop_config()
    protected_cfg["runtime"]["epochs"] = 3
    protected_cfg["runtime"]["seed"] = seed
    protected_cfg["runtime"]["dataset"] = dataset
    protected_result = run_closed_loop_minimization(protected_cfg)
    protected_worst = max(protected_result["history"], key=lambda item: item["metrics"].risk_score)
    protected_metrics = protected_worst["metrics"]

    # compute leakage exposure (excludes epsilon cost for fair comparison)
    baseline_leakage = _leakage_from_item(baseline_worst)
    protected_leakage = _leakage_from_item(protected_worst)

    # Summary table
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

    # Per-epoch deltas (protected minus baseline)
    print("\nPer-epoch counterfactual deltas (protected - baseline):")
    print(f"{'Epoch':<8} {'dAUC':>10} {'dAdv':>10} {'dLeak':>10}")
    for epoch_idx, (b_item, p_item) in enumerate(zip(baseline_result["history"], protected_result["history"]), start=1):
        b_leak = _leakage_from_item(b_item)
        p_leak = _leakage_from_item(p_item)
        print(
            f"{epoch_idx:<8} "
            f"{(p_item['metrics'].mia_auc - b_item['metrics'].mia_auc):>10.3f} "
            f"{(p_item['metrics'].mia_advantage - b_item['metrics'].mia_advantage):>10.3f} "
            f"{(p_leak - b_leak):>10.3f}"
        )
