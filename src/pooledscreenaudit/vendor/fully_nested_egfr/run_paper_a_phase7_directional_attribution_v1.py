#!/usr/bin/env python3
"""Phase 7 outcome-free direction construction and predictive attribution."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("error", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_dms_b15_ablation_v2 import (  # noqa: E402
    ALPHAS,
    CONDS,
    feature_blocks,
    outer_splits,
    wmae,
)
from run_v3_dms_structure_technical_models import load_enriched  # noqa: E402


OUT = ROOT / "outputs/paper_a_phase7_directional_attribution_2026-09-03_v1_1"
FIG = OUT / "figures"
FREEZE = ROOT / "docs/PAPER_A_PHASE7_DIRECTIONAL_ATTRIBUTION_FREEZE_v1_2026-09-03.md"
PHASE6 = ROOT / "outputs/paper_a_phase6_scientific_audit_2026-09-03_v1_3"
ATLAS = ROOT / "outputs/freeze_amendment_v3_2026-08-26/outcome_blind_atlases/dmso_site_technical_qc_atlas.csv"
MATERIAL = 0.02
PARAMETERIZATION_TOL = 0.005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_position(residues: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"wildtype_aa": "A", "amino_acid": "A", "residue": residues})
    return feature_blocks(frame)[2]


def fit_t_standardization(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(t, axis=0)
    med[~np.isfinite(med)] = 0.0
    filled = np.where(np.isfinite(t), t, med)
    center = filled.mean(axis=0)
    scale = filled.std(axis=0)
    keep = scale > 1e-12
    return med, center, np.where(keep, scale, 1.0)


def apply_t_standardization(t: np.ndarray, med: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.where(np.isfinite(t), t, med) - center) / scale


def direction_basis(residues: np.ndarray, t_raw: np.ndarray) -> dict[str, np.ndarray]:
    p = raw_position(residues)
    p_center = p.mean(axis=0)
    pc = p - p_center
    med, tc, ts = fit_t_standardization(t_raw)
    t = apply_t_standardization(t_raw, med, tc, ts)
    control = np.column_stack([np.ones(len(residues)), t])
    u, s, _ = np.linalg.svd(control, full_matrices=False)
    tol = max(control.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
    rank = int(np.sum(s > tol))
    q = u[:, :rank]
    qtp = np.einsum("ij,ik->jk", q, pc, optimize=False)
    residual = pc - np.einsum("ij,jk->ik", q, qtp, optimize=False)
    a = np.einsum("ij,ik->jk", pc, residual, optimize=False)
    b = np.einsum("ij,ik->jk", pc, pc, optimize=False)
    bvals, bvecs = np.linalg.eigh(b)
    keep = bvals > max(float(bvals.max()), 1.0) * 1e-12
    whitening = np.einsum("ij,jk->ik", bvecs[:, keep], np.diag(1.0 / np.sqrt(bvals[keep])), optimize=False)
    core = np.einsum("ij,jk,kl->il", whitening.T, a, whitening, optimize=False)
    vals, vecs = np.linalg.eigh((core + core.T) / 2.0)
    order = np.argsort(vals)[::-1]
    unique = np.clip(vals[order], 0.0, 1.0)
    coefficients = np.einsum("ij,jk->ik", whitening, vecs[:, order], optimize=False)
    signals = np.einsum("ij,jk->ik", pc, coefficients, optimize=False)
    for j in range(signals.shape[1]):
        corr = np.corrcoef(signals[:, j], residues)[0, 1]
        if (np.isfinite(corr) and corr < 0) or (not np.isfinite(corr) and coefficients[np.argmax(np.abs(coefficients[:, j])), j] < 0):
            coefficients[:, j] *= -1
            signals[:, j] *= -1
    gram = np.einsum("ij,ik->jk", signals, signals, optimize=False)
    if np.max(np.abs(gram - np.eye(3))) > 1e-8:
        raise RuntimeError("Direction normalization failed")
    return {
        "p_center": p_center,
        "coefficients": coefficients,
        "unique": unique,
        "shared": 1.0 - unique,
        "signals": signals,
        "control_rank": np.array([rank]),
    }


def direction_values(residues: np.ndarray, basis: dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum("ij,jk->ik", raw_position(residues) - basis["p_center"], basis["coefficients"], optimize=False)


def stable_ridge_from_design(train: pd.DataFrame, x: np.ndarray, xt: np.ndarray, alpha: float) -> np.ndarray:
    y = train.y.to_numpy(float)
    sqrt_weight = np.sqrt(train.weight.to_numpy(float))
    xd = np.column_stack([np.ones(len(x)), x])
    td = np.column_stack([np.ones(len(xt)), xt])
    xw = xd * sqrt_weight[:, None]
    yw = y * sqrt_weight
    gram = np.einsum("ni,nj->ij", xw, xw, optimize=False)
    rhs = np.einsum("ni,n->i", xw, yw, optimize=False)
    penalty = np.eye(xd.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, rhs)
    pred = np.einsum("ni,i->n", td, beta, optimize=False)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Non-finite stable ridge prediction")
    return pred


def build_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tcols: list[str],
    directions_train: np.ndarray,
    directions_test: np.ndarray,
    subset: tuple[int, ...],
    include_t: bool,
    original_p: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    joined = pd.concat([train, test], ignore_index=True)
    s, identity, p = feature_blocks(joined)
    n = len(train)
    blocks = [s, identity]
    if include_t:
        blocks.append(joined[tcols].to_numpy(float))
    if original_p:
        blocks.append(p)
    elif subset:
        blocks.append(np.vstack([directions_train[:, subset], directions_test[:, subset]]))
    z = np.hstack(blocks)
    med = np.nanmedian(z[:n], axis=0)
    med[~np.isfinite(med)] = 0.0
    z = np.where(np.isfinite(z), z, med)
    center = z[:n].mean(axis=0)
    scale = z[:n].std(axis=0)
    scale[scale == 0] = 1.0
    z = (z - center) / scale
    cond = np.column_stack([(joined.condition == c).to_numpy(float) for c in CONDS])
    interactions = np.einsum("ij,ik->ijk", z, cond).reshape(len(z), -1)
    x = np.hstack([cond, z, interactions])
    return x[:n], x[n:]


def tune_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tcols: list[str],
    dir_train: np.ndarray,
    dir_test: np.ndarray,
    subset: tuple[int, ...],
    include_t: bool,
    original_p: bool = False,
) -> tuple[float, np.ndarray]:
    positions = np.array(sorted(train.residue.unique()))
    folds = np.array_split(positions, 3)
    losses = {float(a): [] for a in ALPHAS}
    for held in folds:
        mask = ~train.residue.isin(held).to_numpy()
        tr, va = train.loc[mask], train.loc[~mask]
        if tr.empty or va.empty:
            continue
        x, xv = build_design(
            tr,
            va,
            tcols,
            dir_train[mask],
            dir_train[~mask],
            subset,
            include_t,
            original_p,
        )
        for alpha in ALPHAS:
            pred = stable_ridge_from_design(tr, x, xv, float(alpha))
            losses[float(alpha)].append(wmae(va.y.to_numpy(), pred, va.weight.to_numpy()))
    alpha = min(losses, key=lambda a: (np.mean(losses[a]) if losses[a] else np.inf, a))
    x, xt = build_design(train, test, tcols, dir_train, dir_test, subset, include_t, original_p)
    return float(alpha), stable_ridge_from_design(train, x, xt, alpha)


def subset_label(subset: tuple[int, ...]) -> str:
    return "empty" if not subset else "+".join(f"v{i+1}" for i in subset)


def all_subsets() -> list[tuple[int, ...]]:
    return [tuple(c) for k in range(4) for c in itertools.combinations(range(3), k)]


def selected_splits(frame: pd.DataFrame):
    for challenge, repeat, test_idx, cold_mask in outer_splits(frame):
        if challenge == "random_variant_fixed_test" or challenge == "position_cold":
            yield challenge, repeat, test_idx, cold_mask


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    frame, _, tcols = load_enriched()
    sites = frame[["residue"] + tcols].drop_duplicates("residue").sort_values("residue").reset_index(drop=True)
    global_basis = direction_basis(sites.residue.to_numpy(float), sites[tcols].to_numpy(float))
    global_map = pd.DataFrame({"residue": sites.residue.astype(int)})
    for j in range(3):
        global_map[f"v{j+1}"] = global_basis["signals"][:, j]
    global_map.to_csv(OUT / "global_direction_values_by_residue.csv", index=False)
    coeff_rows = []
    for j in range(3):
        coeff_rows.append({
            "direction": f"v{j+1}",
            "shared_fraction": float(global_basis["shared"][j]),
            "unique_fraction": float(global_basis["unique"][j]),
            "coefficient_scaled_residue": float(global_basis["coefficients"][0, j]),
            "coefficient_scaled_residue_squared": float(global_basis["coefficients"][1, j]),
            "coefficient_scaled_residue_cubed": float(global_basis["coefficients"][2, j]),
        })
    coefficients = pd.DataFrame(coeff_rows)
    coefficients.to_csv(OUT / "global_direction_coefficients.csv", index=False)

    metrics_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    subsets = all_subsets()
    for challenge, repeat, test_idx, cold_mask in selected_splits(frame):
        test = frame.loc[test_idx].copy()
        train = frame.loc[~cold_mask].drop(index=test_idx, errors="ignore").copy()
        train_site = frame.loc[frame.index.isin(train.index), ["residue"] + tcols].drop_duplicates("residue").sort_values("residue")
        fold_basis = direction_basis(train_site.residue.to_numpy(float), train_site[tcols].to_numpy(float))
        common_res = train_site.residue.to_numpy(float)
        global_common = direction_values(common_res, global_basis)
        fold_common = direction_values(common_res, fold_basis)
        row = {"challenge": challenge, "repeat": repeat, "training_positions": int(len(common_res))}
        for j in range(3):
            row[f"global_shared_v{j+1}"] = float(global_basis["shared"][j])
            row[f"fold_shared_v{j+1}"] = float(fold_basis["shared"][j])
            row[f"abs_correlation_v{j+1}"] = float(abs(np.corrcoef(global_common[:, j], fold_common[:, j])[0, 1]))
        qg, _ = np.linalg.qr(global_common - global_common.mean(axis=0))
        qf, _ = np.linalg.qr(fold_common - fold_common.mean(axis=0))
        cross_basis = np.einsum("ij,ik->jk", qg[:, :3], qf[:, :3], optimize=False)
        singular = np.linalg.svd(cross_basis, compute_uv=False)
        row["max_principal_angle_degrees"] = float(np.degrees(np.arccos(np.clip(singular.min(), -1, 1))))
        stability_rows.append(row)

        bases = {"global": global_basis, "fold_fitted": fold_basis}
        for basis_mode, basis in bases.items():
            dtrain = direction_values(train.residue.to_numpy(float), basis)
            dtest = direction_values(test.residue.to_numpy(float), basis)
            for include_t in (False, True):
                for subset in subsets:
                    alpha, pred = tune_and_predict(train, test, tcols, dtrain, dtest, subset, include_t)
                    base = {
                        "challenge": challenge,
                        "repeat": repeat,
                        "basis_mode": basis_mode,
                        "include_T": int(include_t),
                        "subset": subset_label(subset),
                        "subset_size": len(subset),
                        "alpha": alpha,
                        "condition": "ALL",
                        "test_rows": len(test),
                        "weighted_mae": wmae(test.y.to_numpy(), pred, test.weight.to_numpy()),
                    }
                    metrics_rows.append(base)
                    for condition, indexes in test.groupby("condition").groups.items():
                        loc = test.index.get_indexer(indexes)
                        metrics_rows.append(base | {
                            "condition": condition,
                            "test_rows": len(loc),
                            "weighted_mae": wmae(test.iloc[loc].y.to_numpy(), pred[loc], test.iloc[loc].weight.to_numpy()),
                        })
            if basis_mode == "global":
                for include_t in (False, True):
                    alpha, pred = tune_and_predict(train, test, tcols, dtrain, dtest, (), include_t, original_p=True)
                    metrics_rows.append({
                        "challenge": challenge,
                        "repeat": repeat,
                        "basis_mode": "original_P",
                        "include_T": int(include_t),
                        "subset": "original_P",
                        "subset_size": 3,
                        "alpha": alpha,
                        "condition": "ALL",
                        "test_rows": len(test),
                        "weighted_mae": wmae(test.y.to_numpy(), pred, test.weight.to_numpy()),
                    })
        print(f"completed {challenge} split {repeat}", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    stability = pd.DataFrame(stability_rows)
    if not np.isfinite(metrics.weighted_mae).all() or not np.isfinite(stability.select_dtypes(include=[np.number])).all().all():
        raise FloatingPointError("Non-finite Phase 7 output")
    metrics.to_csv(OUT / "direction_subset_model_metrics.csv", index=False)
    stability.to_csv(OUT / "fold_fitted_direction_stability.csv", index=False)

    allm = metrics.query("condition == 'ALL' and basis_mode != 'original_P'").copy()
    base = allm.query("subset == 'empty'")[["challenge", "repeat", "basis_mode", "include_T", "weighted_mae"]].rename(columns={"weighted_mae": "baseline_mae"})
    gains = allm.merge(base, on=["challenge", "repeat", "basis_mode", "include_T"], validate="many_to_one")
    gains["gain_vs_matched_baseline"] = gains.baseline_mae - gains.weighted_mae
    gains.to_csv(OUT / "direction_subset_gains.csv", index=False)

    shapley_rows = []
    for keys, group in gains.groupby(["challenge", "repeat", "basis_mode", "include_T"], sort=False):
        value = {row.subset: row.gain_vs_matched_baseline for row in group.itertuples()}
        for i in range(3):
            phi = 0.0
            others = [j for j in range(3) if j != i]
            for k in range(3):
                for comb in itertools.combinations(others, k):
                    s0 = tuple(sorted(comb))
                    s1 = tuple(sorted(comb + (i,)))
                    weight = math.factorial(k) * math.factorial(2-k) / math.factorial(3)
                    phi += weight * (value[subset_label(s1)] - value[subset_label(s0)])
            shapley_rows.append({
                "challenge": keys[0], "repeat": keys[1], "basis_mode": keys[2], "include_T": keys[3],
                "direction": f"v{i+1}", "shapley_gain": phi,
                "full_direction_gain": value["v1+v2+v3"],
            })
    shapley = pd.DataFrame(shapley_rows)
    shapley["efficiency_error"] = shapley.groupby(["challenge", "repeat", "basis_mode", "include_T"]).shapley_gain.transform("sum") - shapley.full_direction_gain
    if shapley.efficiency_error.abs().max() > 1e-10:
        raise RuntimeError("Shapley efficiency failed")
    shapley.to_csv(OUT / "direction_shapley_by_split.csv", index=False)

    reference = metrics.query("condition == 'ALL' and basis_mode == 'original_P'").pivot(index=["challenge", "repeat"], columns="include_T", values="weighted_mae").reset_index().rename(columns={0: "original_P_mae", 1: "T_plus_original_P_mae"})
    baseref = gains.query("basis_mode == 'global' and subset == 'empty'").pivot(index=["challenge", "repeat"], columns="include_T", values="baseline_mae").reset_index().rename(columns={0: "base_mae", 1: "T_base_mae"})
    fullref = gains.query("basis_mode == 'global' and subset == 'v1+v2+v3'").pivot(index=["challenge", "repeat"], columns="include_T", values="weighted_mae").reset_index().rename(columns={0: "direction_full_mae", 1: "T_plus_direction_full_mae"})
    parameterization = reference.merge(baseref, on=["challenge", "repeat"]).merge(fullref, on=["challenge", "repeat"])
    parameterization["original_P_gain"] = parameterization.base_mae - parameterization.original_P_mae
    parameterization["direction_full_gain"] = parameterization.base_mae - parameterization.direction_full_mae
    parameterization["gain_difference_direction_minus_original"] = parameterization.direction_full_gain - parameterization.original_P_gain
    parameterization["original_P_given_T_gain"] = parameterization.T_base_mae - parameterization.T_plus_original_P_mae
    parameterization["direction_full_given_T_gain"] = parameterization.T_base_mae - parameterization.T_plus_direction_full_mae
    parameterization.to_csv(OUT / "direction_parameterization_reference.csv", index=False)

    primary_g = gains.query("challenge == 'random_variant_fixed_test' and basis_mode == 'global'")
    primary_s = shapley.query("challenge == 'random_variant_fixed_test' and basis_mode == 'global'")
    no_t = primary_g.query("include_T == 0").groupby("subset").gain_vs_matched_baseline.mean()
    with_t = primary_g.query("include_T == 1").groupby("subset").gain_vs_matched_baseline.mean()
    phi = primary_s.query("include_T == 0").groupby("direction").shapley_gain.mean()
    full_gain = float(no_t["v1+v2+v3"])
    v3_share = float(phi["v3"] / full_gain) if full_gain != 0 else float("nan")
    primary_stability = stability.query("challenge == 'random_variant_fixed_test'")
    pdiff = parameterization.query("challenge == 'random_variant_fixed_test'").gain_difference_direction_minus_original
    gates = {
        "v3_shapley_share_ge_0_70": bool(np.isfinite(v3_share) and v3_share >= 0.70),
        "v1_v2_combined_gain_lt_0_02": bool(no_t["v1+v2"] < MATERIAL),
        "v1_standalone_gain_lt_0_02": bool(no_t["v1"] < MATERIAL),
        "v2_standalone_gain_lt_0_02": bool(no_t["v2"] < MATERIAL),
        "v3_given_T_gain_lt_0_02": bool(with_t["v3"] < MATERIAL),
        "v3_material_in_at_least_8_of_10": bool((primary_g.query("include_T == 0 and subset == 'v3'").gain_vs_matched_baseline >= MATERIAL).sum() >= 8),
        "fold_fitted_v3_median_abs_correlation_ge_0_90": bool(primary_stability.abs_correlation_v3.median() >= 0.90),
        "direction_vs_original_mean_gain_difference_abs_le_0_005": bool(abs(float(pdiff.mean())) <= PARAMETERIZATION_TOL),
    }
    decision = "PREDICTIVE_INFORMATION_CONCENTRATED_IN_MOST_SHARED_DIRECTION" if all(gates.values()) else "DIRECTIONAL_CONCENTRATION_NOT_ESTABLISHED"
    summary = {
        "status": "PASS",
        "timing": "post-Phase-6 hypothesis-driven exploratory; frozen before direction-level outcomes",
        "decision": decision,
        "global_shared_fractions": {f"v{i+1}": float(global_basis["shared"][i]) for i in range(3)},
        "primary_mean_gains_without_T": {str(k): float(v) for k, v in no_t.items()},
        "primary_mean_gains_with_T": {str(k): float(v) for k, v in with_t.items()},
        "primary_mean_shapley_without_T": {str(k): float(v) for k, v in phi.items()},
        "v3_shapley_share_of_full_direction_gain": v3_share,
        "v3_material_primary_folds": int((primary_g.query("include_T == 0 and subset == 'v3'").gain_vs_matched_baseline >= MATERIAL).sum()),
        "fold_fitted_v3_median_abs_correlation": float(primary_stability.abs_correlation_v3.median()),
        "direction_vs_original_mean_gain_difference": float(pdiff.mean()),
        "decision_gates": gates,
        "claim_boundary": "Representation-level localization only; no technical mechanism or causal shortcut claim.",
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics, gains, shapley, summary, stability, coefficients


def provenance_table() -> None:
    rows = [
        {"resource": "Source article", "locator": "https://doi.org/10.1038/s41698-025-01086-2", "access_date": "2026-09-03", "result": "Methods reported orientation-specific forward/reverse oligo pools and two barcoded sequencing subamplicons", "failure_or_limit": "No public per-variant library-synthesis tile map", "fallback": "Evaluate source-defined sequencing partition and label its scope", "author_contact": "Not performed"},
        {"resource": "NCBI transcript", "locator": "NM_005228.5 via E-utilities efetch", "access_date": "2026-09-03", "result": "Primer sequences aligned; shared primer spans approximately codons 846-852", "failure_or_limit": "Primer span does not prove synthesis batches", "fallback": "Use only to define sequencing-boundary sensitivity", "author_contact": "Not applicable"},
        {"resource": "NCBI GEO", "locator": "GSE305057 full SOFT", "access_date": "2026-09-03", "result": "Eight samples and public RAW/SRA relations; no selected tile/lane/PCR-batch field", "failure_or_limit": "Initial timed fetch and an intermediate zero-sample parse were retained in v1_1/v1_2 audit directories", "fallback": "Successful full-SOFT parse in Phase 6 v1_3", "author_contact": "Not performed"},
        {"resource": "Public metadata conclusion", "locator": "source_partition_audit.json", "access_date": "2026-09-03", "result": "Sequencing subamplicons available; library-synthesis tiles unavailable", "failure_or_limit": "Absence from inspected public metadata is not proof the metadata never existed", "fallback": "Prohibit verified-tile claims", "author_contact": "No message sent; no reply represented"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "Supplementary_Table_S5_source_partition_provenance.csv", index=False)


def plot_summary(gains: pd.DataFrame, shapley: pd.DataFrame, summary: dict[str, object], stability: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    spectrum = pd.read_csv(PHASE6 / "phase6_rank_aware_geometry_spectrum.csv")
    direction_values_df = pd.read_csv(OUT / "global_direction_values_by_residue.csv")
    primary = gains.query("challenge == 'random_variant_fixed_test' and basis_mode == 'global'")
    order = ["v1", "v2", "v3", "v1+v2", "v1+v3", "v2+v3", "v1+v2+v3"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    ax = axes[0, 0]
    for name, label, color in [("T_only", "T only", "#0072B2"), ("source_subamplicon_only", "2 subamplicons", "#009E73"), ("analyst_equal_width_blocks_only", "20-aa bins", "#D55E00")]:
        q = spectrum.query("design == @name")
        if name == "analyst_equal_width_blocks_only":
            q = q.query("width_aa == 20 and origin_offset_aa == 0")
        q = q.sort_values("ordered_position_direction")
        ax.plot([1, 2, 3], 100*q.shared_variance_fraction.to_numpy(), marker="o", label=label, color=color)
    ax.set(title="A  Outcome-free shared spectrum", xlabel="Direction", ylabel="Shared variance (%)", xticks=[1,2,3], ylim=(-3,103)); ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    data = [primary.query("include_T == 0 and subset == @s").gain_vs_matched_baseline.to_numpy() for s in ["v1","v2","v3"]]
    ax.boxplot(data, labels=["v1","v2","v3"], showfliers=False)
    for i, vals in enumerate(data, 1): ax.scatter(np.full(len(vals), i), vals, s=11, color="#444444", alpha=.7)
    ax.axhline(MATERIAL, ls="--", lw=1, color="#B2182B"); ax.set(title="B  Stand-alone direction gain", ylabel="MAE gain")

    ax = axes[0, 2]
    ps = shapley.query("challenge == 'random_variant_fixed_test' and basis_mode == 'global' and include_T == 0")
    means = ps.groupby("direction").shapley_gain.mean().reindex(["v1","v2","v3"])
    ax.bar(means.index, means.values, color=["#56B4E9","#E69F00","#8E5EA2"]); ax.axhline(0,color="black",lw=.6)
    ax.set(title="C  Exact Shapley allocation", ylabel="Allocated MAE gain")

    ax = axes[1, 0]
    cond = primary.query("include_T == 1 and subset in ['v1','v2','v3']")
    for i, name in enumerate(["v1","v2","v3"],1):
        vals=cond.query("subset == @name").gain_vs_matched_baseline.to_numpy(); ax.scatter(np.full(len(vals),i),vals,s=12,alpha=.7)
        ax.hlines(vals.mean(),i-.25,i+.25,color="black",lw=2)
    ax.axhline(MATERIAL,ls="--",lw=1,color="#B2182B"); ax.set(title="D  Direction gain after T", ylabel="Conditional MAE gain", xticks=[1,2,3], xticklabels=["v1","v2","v3"])

    ax = axes[1, 1]
    for j,color in enumerate(["#56B4E9","#E69F00","#8E5EA2"]):
        values=direction_values_df[f"v{j+1}"]; z=(values-values.mean())/values.std(); shared=100*summary["global_shared_fractions"][f"v{j+1}"]
        ax.plot(direction_values_df.residue,z,label=f"v{j+1}: {shared:.1f}% shared",color=color)
    ax.set(title="E  Frozen position directions", xlabel="EGFR residue", ylabel="Standardized direction"); ax.legend(frameon=False,fontsize=7)

    ax = axes[1, 2]
    st=stability.query("challenge == 'random_variant_fixed_test'")
    ax.scatter(st.repeat,st.abs_correlation_v3,color="#8E5EA2",label="|corr| global vs fold v3")
    ax.axhline(.90,ls="--",color="#B2182B",lw=1); ax.set(title="F  Fold-fitted v3 stability",xlabel="Primary split",ylabel="Absolute correlation",ylim=(0,1.03)); ax.legend(frameon=False,fontsize=7)
    fig.suptitle("Phase 7 direction-level predictive attribution", weight="bold")
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(FIG / "Figure_directional_attribution_phase7.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Figure_directional_attribution_phase7.tif", dpi=600, pil_kwargs={"compression":"tiff_lzw"}, bbox_inches="tight")
    fig.savefig(FIG / "Figure_directional_attribution_phase7_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def manifests() -> None:
    inputs = [FREEZE, Path(__file__), ATLAS, PHASE6 / "phase6_rank_aware_geometry_spectrum.csv", PHASE6 / "cross_replicate_fold_gains_reused.csv", PHASE6 / "count_generator_S2_threshold_audit.csv", PHASE6 / "source_partition_audit.json"]
    with (OUT / "input_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["path","sha256","bytes"]); writer.writeheader()
        writer.writerows([{"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"bytes":p.stat().st_size} for p in inputs])
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="output_manifest_sha256.csv" and not p.name.startswith("._")]
    with (OUT / "output_manifest_sha256.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["path","sha256","bytes"]); writer.writeheader()
        writer.writerows([{"path":str(p.relative_to(OUT)),"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(files)])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    FIG.mkdir(parents=True)
    metrics, gains, shapley, summary, stability, coefficients = evaluate()
    provenance_table()
    pd.read_csv(PHASE6 / "cross_replicate_fold_gains_reused.csv").to_csv(OUT / "Supplementary_Table_S4_cross_replicate_fold_results.csv", index=False)
    plot_summary(gains, shapley, summary, stability, coefficients)
    environment = {"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__}
    (OUT / "environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    manifests()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
