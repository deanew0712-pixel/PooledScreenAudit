#!/usr/bin/env python3
"""Run v3 DMS position/structure/technical models and hinge diagnostics."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_dms_b15_ablation_v2 import (
    ALPHAS,
    CONDS,
    b0,
    choose_one_edge,
    feature_blocks,
    load_frame,
    outer_splits,
    ridge_from_design,
    sha,
    wmae,
)


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "outputs" / "freeze_amendment_v3_2026-08-26" / "outcome_blind_atlases"
OUT = ROOT / "outputs" / "freeze_amendment_v3_2026-08-26" / "structure_technical_models"
STRUCTURE = ATLAS / "continuous_structure_feature_atlas.csv"
TECHNICAL = ATLAS / "dmso_site_technical_qc_atlas.csv"

MODEL_BLOCKS = {
    "B175": ("S", "I"),
    "B2_pos": ("S", "I", "P"),
    "B2_struct": ("S", "I", "R"),
    "B2_hybrid": ("S", "I", "P", "R"),
    "B0_T": ("T",),
    "B0_T_P": ("T", "P"),
    "B175_T": ("S", "I", "T"),
    "B175_T_P": ("S", "I", "T", "P"),
}


def load_enriched() -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = load_frame()
    structure = pd.read_csv(STRUCTURE)
    technical = pd.read_csv(TECHNICAL)
    frame = frame.merge(structure, on="residue", validate="many_to_one")
    frame = frame.merge(technical, on=["site", "residue"], validate="many_to_one")
    rcols = [c for c in structure.columns if c != "residue"]
    tcols = [c for c in technical.columns if c not in {"site", "residue", "wildtype_codon"}]
    return frame, rcols, tcols


def raw_blocks(frame: pd.DataFrame, rcols: list[str], tcols: list[str], position_override=None):
    s, i, p = feature_blocks(frame, context_residue=position_override)
    r = frame[rcols].to_numpy(float)
    t = frame[tcols].to_numpy(float)
    return {"S": s, "I": i, "P": p, "R": r, "T": t}


def design_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    blocks: tuple[str, ...],
    rcols: list[str],
    tcols: list[str],
    train_position=None,
    test_position=None,
):
    joined = pd.concat([train, test], ignore_index=True)
    position_override = None
    if train_position is not None:
        position_override = np.concatenate([np.asarray(train_position), np.asarray(test_position if test_position is not None else test.residue)])
    raw = raw_blocks(joined, rcols, tcols, position_override)
    z = np.hstack([raw[b] for b in blocks])
    n = len(train)
    # Train-only median imputation and scaling.
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


def tune(train: pd.DataFrame, model: str, rcols: list[str], tcols: list[str]) -> float:
    blocks = MODEL_BLOCKS[model]
    positions = np.array(sorted(train.residue.unique()))
    folds = np.array_split(positions, 3)
    losses = {a: [] for a in ALPHAS}
    for held in folds:
        mask = ~train.residue.isin(held).to_numpy()
        tr, va = train.loc[mask], train.loc[~mask]
        if tr.empty or va.empty:
            continue
        x, xt = design_model(tr, va, blocks, rcols, tcols)
        for alpha in ALPHAS:
            pred = ridge_from_design(tr, x, xt, alpha)
            losses[alpha].append(wmae(va.y.to_numpy(), pred, va.weight.to_numpy()))
    return float(min(ALPHAS, key=lambda a: (np.mean(losses[a]) if losses[a] else np.inf, a)))


def predict(train, test, model, alpha, rcols, tcols, train_position=None, test_position=None):
    x, xt = design_model(train, test, MODEL_BLOCKS[model], rcols, tcols, train_position, test_position)
    return ridge_from_design(train, x, xt, alpha)


def all_splits(frame: pd.DataFrame):
    yield from outer_splits(frame)
    block_number = 0
    for start in range(712, 980, 20):
        stop = min(start + 19, 979)
        mask = frame.residue.between(start, stop).to_numpy()
        held = frame.loc[mask, "variant"].unique()
        block_number += 1
        yield f"continuous_block_{start}_{stop}", 1, choose_one_edge(frame, held, 20263000 + block_number), mask


def evaluate(frame, rcols, tcols):
    rows = []
    for challenge, repeat, test_idx, cold_mask in all_splits(frame):
        test = frame.loc[test_idx].copy()
        train = frame.loc[~cold_mask].drop(index=test_idx, errors="ignore").copy()
        for model in MODEL_BLOCKS:
            alpha = tune(train, model, rcols, tcols)
            pred = predict(train, test, model, alpha, rcols, tcols)
            base = {
                "challenge": challenge,
                "repeat": repeat,
                "model": model,
                "alpha": alpha,
                "test_rows": len(test),
                "test_positions": test.residue.nunique(),
                "weighted_mae": wmae(test.y.to_numpy(), pred, test.weight.to_numpy()),
            }
            rows.append(base | {"condition": "ALL"})
            for condition, idx in test.groupby("condition").groups.items():
                loc = test.index.get_indexer(idx)
                if len(loc):
                    rows.append(base | {"condition": condition, "test_rows": len(loc), "weighted_mae": wmae(test.iloc[loc].y.to_numpy(), pred[loc], test.iloc[loc].weight.to_numpy())})
    return pd.DataFrame(rows)


def contrasts(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    allm = metrics[metrics.condition.eq("ALL")].pivot(index=["challenge", "repeat"], columns="model", values="weighted_mae").reset_index()
    allm["delta_pos_vs_b175"] = allm.B2_pos - allm.B175
    allm["delta_struct_vs_b175"] = allm.B2_struct - allm.B175
    allm["delta_hybrid_vs_pos"] = allm.B2_hybrid - allm.B2_pos
    allm["delta_hybrid_vs_struct"] = allm.B2_hybrid - allm.B2_struct
    allm["g0_position_gain"] = allm.B175 - allm.B2_pos
    allm["gT_position_gain_after_technical"] = allm.B175_T - allm.B175_T_P
    allm["technical_explained_fraction_unbounded"] = np.where(
        allm.g0_position_gain > 0,
        1 - allm.gT_position_gain_after_technical / allm.g0_position_gain,
        np.nan,
    )
    summary = allm.groupby("challenge", sort=False).agg(
        splits=("repeat", "size"),
        delta_pos_vs_b175=("delta_pos_vs_b175", "mean"),
        delta_struct_vs_b175=("delta_struct_vs_b175", "mean"),
        delta_hybrid_vs_pos=("delta_hybrid_vs_pos", "mean"),
        delta_hybrid_vs_struct=("delta_hybrid_vs_struct", "mean"),
        g0_position_gain=("g0_position_gain", "mean"),
        gT_position_gain_after_technical=("gT_position_gain_after_technical", "mean"),
        technical_explained_fraction_median=("technical_explained_fraction_unbounded", "median"),
    ).reset_index()
    return allm, summary


def hinge_diagnostics(frame, rcols, tcols):
    hinge = frame.residue.between(792, 796)
    train, test = frame.loc[~hinge].copy(), frame.loc[hinge].copy()
    pred = {}
    alphas = {}
    for model in ["B175", "B2_pos", "B2_struct", "B2_hybrid"]:
        alphas[model] = tune(train, model, rcols, tcols)
        pred[model] = predict(train, test, model, alphas[model], rcols, tcols)
    row_pred = test[["condition", "residue", "variant", "y", "weight"]].copy()
    for model, values in pred.items():
        row_pred[f"pred_{model}"] = values
        row_pred[f"abs_error_{model}"] = np.abs(row_pred.y - values)

    per_position = []
    for (residue, condition), g in row_pred.groupby(["residue", "condition"]):
        row = {"residue": residue, "condition": condition, "edges": len(g)}
        for model in pred:
            row[f"mae_{model}"] = float(np.average(g[f"abs_error_{model}"], weights=g.weight))
        row["delta_pos_vs_b175"] = row["mae_B2_pos"] - row["mae_B175"]
        row["delta_struct_vs_b175"] = row["mae_B2_struct"] - row["mae_B175"]
        row["delta_hybrid_vs_b175"] = row["mae_B2_hybrid"] - row["mae_B175"]
        per_position.append(row)
    per_position = pd.DataFrame(per_position)

    jackknife = []
    for omitted in sorted(test.residue.unique()):
        keep = row_pred.residue.ne(omitted)
        row = {"omitted_residue": omitted, "remaining_positions": int(row_pred.loc[keep, "residue"].nunique())}
        for model in pred:
            row[f"mae_{model}"] = float(np.average(row_pred.loc[keep, f"abs_error_{model}"], weights=row_pred.loc[keep, "weight"]))
        row["delta_pos_vs_b175"] = row["mae_B2_pos"] - row["mae_B175"]
        row["delta_struct_vs_b175"] = row["mae_B2_struct"] - row["mae_B175"]
        row["delta_hybrid_vs_b175"] = row["mae_B2_hybrid"] - row["mae_B175"]
        jackknife.append(row)
    jackknife = pd.DataFrame(jackknife)

    ref_mae = wmae(test.y.to_numpy(), pred["B175"], test.weight.to_numpy())
    observed_delta = wmae(test.y.to_numpy(), pred["B2_pos"], test.weight.to_numpy()) - ref_mae
    positions = np.array([792, 793, 794, 795, 796])
    perm_rows = []
    for order in itertools.permutations(positions.tolist()):
        mapping = dict(zip(positions, order))
        mapped = test.residue.map(mapping).to_numpy(float)
        pp = predict(train, test, "B2_pos", alphas["B2_pos"], rcols, tcols, test_position=mapped, train_position=train.residue.to_numpy(float))
        delta = wmae(test.y.to_numpy(), pp, test.weight.to_numpy()) - ref_mae
        perm_rows.append({"mapping": ";".join(f"{a}->{b}" for a, b in zip(positions, order)), "delta_pos_vs_b175": delta, "is_observed_mapping": int(np.array_equal(positions, np.array(order)))})
    exact = pd.DataFrame(perm_rows)
    exact_summary = {
        "observed_delta_pos_vs_b175": observed_delta,
        "permutations": len(exact),
        "lower_tail_fraction_including_observed": float(np.mean(exact.delta_pos_vs_b175 <= observed_delta + 1e-15)),
        "upper_tail_fraction_including_observed": float(np.mean(exact.delta_pos_vs_b175 >= observed_delta - 1e-15)),
        "min": float(exact.delta_pos_vs_b175.min()),
        "median": float(exact.delta_pos_vs_b175.median()),
        "max": float(exact.delta_pos_vs_b175.max()),
        "interpretation_scope": "descriptive exact mapping sensitivity over five hinge positions; not a population CI",
    }
    return row_pred, per_position, jackknife, exact, exact_summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    frame, rcols, tcols = load_enriched()
    metrics = evaluate(frame, rcols, tcols)
    paired, summary = contrasts(metrics)
    metrics.to_csv(OUT / "model_metrics_by_challenge_condition.csv", index=False)
    paired.to_csv(OUT / "paired_model_contrasts_by_split.csv", index=False)
    summary.to_csv(OUT / "model_contrast_summary.csv", index=False)
    row_pred, per_position, jackknife, exact, exact_summary = hinge_diagnostics(frame, rcols, tcols)
    row_pred.to_csv(OUT / "hinge_row_predictions.csv", index=False)
    per_position.to_csv(OUT / "hinge_position_condition_metrics.csv", index=False)
    jackknife.to_csv(OUT / "hinge_leave_one_position_out.csv", index=False)
    exact.to_csv(OUT / "hinge_exact_5factorial_position_mappings.csv", index=False)
    (OUT / "hinge_exact_summary.json").write_text(json.dumps(exact_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = summary.loc[summary.challenge.eq("random_variant_fixed_test")].iloc[0]
    ep = float(primary.technical_explained_fraction_median)
    technical_decision = "MOSTLY_TECHNICAL" if ep >= .70 else "MIXED" if ep >= .30 else "LIMITED_TECHNICAL_EXPLANATION"
    manifest = {
        "status": "COMPLETED",
        "structure_atlas_sha256": sha(STRUCTURE),
        "technical_atlas_sha256": sha(TECHNICAL),
        "models": MODEL_BLOCKS,
        "primary_challenge": "random_variant_fixed_test",
        "primary_delta_pos_vs_b175": float(primary.delta_pos_vs_b175),
        "primary_delta_struct_vs_b175": float(primary.delta_struct_vs_b175),
        "primary_delta_hybrid_vs_pos": float(primary.delta_hybrid_vs_pos),
        "primary_technical_explained_fraction_median": ep,
        "technical_decision": technical_decision,
        "prior_outputs_overwritten": False,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(exact_summary, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

