#!/usr/bin/env python3
"""Frozen DMS B1.5 ablation with an identifiable position-block null.

The protocol's B2-B1.5 contrast mixes absolute amino-acid identity and residue
position.  This implementation therefore inserts B1.75 (B1.5 + WT/Mut
identity) and calibrates only B2-B1.75 with a position-level permutation null.
Prior outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs" / "two_paper_upgrade_2026-08-25_v2" / "dms_count_aware_v1_1" / "dms_count_aware_labels_v1_1.csv"
ASSIGN = ROOT / "outputs" / "two_paper_upgrade_2026-08-25_v2" / "dms_region_cold" / "dms_leave_set_assignments.csv"
OUT = ROOT / "outputs" / "freeze_amendment_v2_2026-08-26" / "dms_b15_ablation"
SEED = 20260826
ALPHAS = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
AA = list("ACDEFGHIKLMNPQRSTVWY")
CONDS = ["BLU945", "osimertinib", "BLU945_plus_osimertinib"]

HYDRO = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -.8, -.7, 4.2, -.9, -1.3]))
VOLUME = dict(zip(AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6]))
CHARGE = {a: 0.0 for a in AA} | {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
POLARITY = dict(zip(AA, [8.1, 5.5, 13.0, 12.3, 5.2, 9.0, 10.4, 5.2, 11.3, 4.9, 5.7, 11.6, 8.0, 10.5, 10.5, 9.2, 8.6, 5.9, 5.4, 6.2]))
AROMATIC = {a: float(a in "FWY") for a in AA}

BLOSUM_ORDER = list("ARNDCQEGHILKMFPSTWYV")
BLOSUM_ROWS = [
    "4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0",
    "-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3",
    "-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3",
    "-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3",
    "0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1",
    "-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2",
    "-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2",
    "0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3",
    "-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3",
    "-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3",
    "-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1",
    "-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2",
    "-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1",
    "-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1",
    "-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2",
    "1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2",
    "0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0",
    "-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3",
    "-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1",
    "0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4",
]
BLOSUM = {(a, b): float(BLOSUM_ROWS[i].split()[j]) for i, a in enumerate(BLOSUM_ORDER) for j, b in enumerate(BLOSUM_ORDER)}

GRANTHAM_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
GRANTHAM_UPPER = {
    "A": [0,195,126,107,113,60,86,94,106,96,84,111,27,91,112,99,58,64,148,112],
    "C": [0,154,170,205,159,174,198,202,198,196,139,169,154,180,112,149,192,215,194],
    "D": [0,45,177,94,81,168,101,172,160,23,108,61,96,65,85,152,181,160],
    "E": [0,140,98,40,134,56,138,126,42,93,29,54,80,65,121,152,122],
    "F": [0,153,100,21,102,22,28,158,114,116,97,155,103,50,40,22],
    "G": [0,98,135,127,138,127,80,42,87,125,56,59,109,184,147],
    "H": [0,94,32,99,87,68,77,24,29,89,47,84,115,83],
    "I": [0,102,5,10,149,95,109,97,142,89,29,61,33],
    "K": [0,107,95,94,103,53,26,121,78,97,110,85],
    "L": [0,15,153,98,113,102,145,92,32,61,36],
    "M": [0,142,87,101,91,135,81,21,67,36],
    "N": [0,91,46,86,46,65,133,174,143],
    "P": [0,76,103,74,38,68,147,110],
    "Q": [0,43,68,42,96,130,99],
    "R": [0,110,71,96,101,77],
    "S": [0,58,124,177,144],
    "T": [0,69,128,92],
    "V": [0,88,55], "W": [0,37], "Y": [0],
}
GRANTHAM = {}
for i, a in enumerate(GRANTHAM_ORDER):
    for off, value in enumerate(GRANTHAM_UPPER[a]):
        b = GRANTHAM_ORDER[i + off]
        GRANTHAM[a, b] = GRANTHAM[b, a] = float(value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wmae(y, pred, weight) -> float:
    return float(np.sum(weight * np.abs(y - pred)) / np.sum(weight))


def feature_blocks(frame: pd.DataFrame, context_residue: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt = frame.wildtype_aa.astype(str).to_numpy()
    mut = frame.amino_acid.astype(str).to_numpy()
    delta = lambda table: np.array([table[m] - table[w] for w, m in zip(wt, mut)], dtype=float)
    b15 = np.column_stack([
        [BLOSUM[w, m] for w, m in zip(wt, mut)],
        [GRANTHAM[w, m] for w, m in zip(wt, mut)],
        delta(HYDRO), delta(VOLUME), delta(CHARGE), delta(POLARITY),
        (mut == "G") & (wt != "G"), (wt == "G") & (mut != "G"),
        (mut == "P") & (wt != "P"), (wt == "P") & (mut != "P"),
        (mut == "C") & (wt != "C"), (wt == "C") & (mut != "C"),
        np.sign(delta(CHARGE)) != 0,
    ]).astype(float)
    identity = np.column_stack([(wt == a).astype(float) for a in AA] + [(mut == a).astype(float) for a in AA] + [delta(AROMATIC)])
    residue = frame.residue.to_numpy(float) if context_residue is None else np.asarray(context_residue, float)
    scaled = (residue - 797.0) / 100.0
    position = np.column_stack([scaled, scaled**2, scaled**3])
    return b15, identity, position


def design(train: pd.DataFrame, test: pd.DataFrame, level: str, train_context=None, test_context=None):
    joined = pd.concat([train, test], ignore_index=True)
    context = None
    if train_context is not None:
        context = np.concatenate([np.asarray(train_context), np.asarray(test_context if test_context is not None else test.residue)])
    b15, identity, position = feature_blocks(joined, context)
    blocks = {"B15_substitution_severity": b15,
              "B175_absolute_aa_identity": np.hstack([b15, identity]),
              "B2_position_context": np.hstack([b15, identity, position])}
    z = blocks[level]
    n = len(train)
    center = z[:n].mean(axis=0)
    scale = z[:n].std(axis=0)
    scale[scale == 0] = 1.0
    z = (z - center) / scale
    cond = np.column_stack([(joined.condition == c).to_numpy(float) for c in CONDS])
    interactions = np.einsum("ij,ik->ijk", z, cond).reshape(len(z), -1)
    x = np.hstack([cond, z, interactions])
    return x[:n], x[n:]


def ridge_from_design(train, x, xt, alpha):
    y = train.y.to_numpy(float)
    weight = np.sqrt(train.weight.to_numpy(float))
    xd = np.column_stack([np.ones(len(x)), x])
    td = np.column_stack([np.ones(len(xt)), xt])
    penalty = np.eye(xd.shape[1]) * alpha
    penalty[0, 0] = 0.0
    xw = xd * weight[:, None]
    gram = xw.T @ xw + penalty
    # The ridge grid is strictly positive, so a direct solve is both stable and
    # substantially faster than recomputing an SVD/pseudoinverse for every fold.
    beta = np.linalg.solve(gram, xw.T @ (y * weight))
    return td @ beta


def ridge(train, test, alpha, level, train_context=None, test_context=None):
    x, xt = design(train, test, level, train_context, test_context)
    return ridge_from_design(train, x, xt, alpha)


def tune(train, level, context=None):
    positions = np.array(sorted(train.residue.unique()))
    folds = np.array_split(positions, 3)
    losses = {a: [] for a in ALPHAS}
    for held in folds:
        tr_mask = ~train.residue.isin(held).to_numpy()
        tr, va = train.loc[tr_mask], train.loc[~tr_mask]
        if tr.empty or va.empty:
            continue
        tr_ctx = context[tr_mask] if context is not None else None
        va_ctx = context[~tr_mask] if context is not None else None
        x, xt = design(tr, va, level, tr_ctx, va_ctx)
        for alpha in ALPHAS:
            pred = ridge_from_design(tr, x, xt, alpha)
            losses[alpha].append(wmae(va.y.to_numpy(), pred, va.weight.to_numpy()))
    return min(ALPHAS, key=lambda a: (np.mean(losses[a]) if losses[a] else np.inf, a))


def b0(train, test):
    means = train.groupby("condition", observed=True).apply(lambda g: np.average(g.y, weights=g.weight), include_groups=False).to_dict()
    fallback = np.average(train.y, weights=train.weight)
    return test.condition.map(means).fillna(fallback).to_numpy(float)


def choose_one_edge(frame, variants, seed):
    rng = np.random.default_rng(seed)
    return np.array([int(rng.choice(frame.index[frame.variant.eq(v)].to_numpy())) for v in sorted(variants)])


def outer_splits(frame):
    variants = np.array(sorted(frame.variant.unique()))
    for rep in range(1, 11):
        rng = np.random.default_rng(20260825 + rep)
        held = rng.choice(variants, int(np.ceil(.2 * len(variants))), replace=False)
        yield "random_variant_fixed_test", rep, choose_one_edge(frame, held, 20260925 + rep), frame.variant.isin(held).to_numpy()
    positions = np.array(sorted(frame.residue.unique()))
    rng = np.random.default_rng(20260825); rng.shuffle(positions)
    for fold, heldpos in enumerate(np.array_split(positions, 5), 1):
        held = frame.loc[frame.residue.isin(heldpos), "variant"].unique()
        yield "position_cold", fold, choose_one_edge(frame, held, 20261025 + fold), frame.residue.isin(heldpos).to_numpy()
    sets = [(f"region_{r}", frame.canonical_region.eq(r).to_numpy()) for r in ["p_loop", "alpha_c_helix", "hinge_motif", "activation_segment"]]
    sets += [(col, frame[col].astype(bool).to_numpy()) for col in ["osimertinib_contact_4A", "blu945_contact_4A", "c797_sequence_window_795_799"]]
    for number, (name, mask) in enumerate(sets, 1):
        held = frame.loc[mask, "variant"].unique()
        yield name, 1, choose_one_edge(frame, held, 20261125 + number), mask


def load_frame():
    labels = pd.read_csv(LABELS).rename(columns={"m2_posterior_mean": "y", "posterior_reliability": "weight"})
    assign = pd.read_csv(ASSIGN)
    return labels.merge(assign, on=["residue", "variant", "wildtype_aa", "amino_acid", "single_nucleotide_accessible"], validate="many_to_one").reset_index(drop=True)


def observed_ablation(frame):
    rows = []
    levels = ["B15_substitution_severity", "B175_absolute_aa_identity", "B2_position_context"]
    for challenge, repeat, test_idx, cold_mask in outer_splits(frame):
        test = frame.loc[test_idx].copy()
        for scheme, train in [("record_leaky", frame.drop(index=test_idx)), ("cold", frame.loc[~cold_mask].drop(index=test_idx, errors="ignore"))]:
            pred0 = b0(train, test)
            rows.append({"challenge": challenge, "repeat": repeat, "scheme": scheme, "model": "B0_condition_only", "alpha": np.nan, "mae": wmae(test.y.to_numpy(), pred0, test.weight.to_numpy()), "train_rows": len(train), "test_rows": len(test)})
            for level in levels:
                alpha = tune(train, level)
                pred = ridge(train, test, alpha, level)
                rows.append({"challenge": challenge, "repeat": repeat, "scheme": scheme, "model": level, "alpha": alpha, "mae": wmae(test.y.to_numpy(), pred, test.weight.to_numpy()), "train_rows": len(train), "test_rows": len(test)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "dms_b15_fold_metrics.csv", index=False)
    cold = metrics[metrics.scheme.eq("cold")].pivot(index=["challenge", "repeat"], columns="model", values="mae").reset_index()
    cold["delta_b15_vs_b0"] = cold.B15_substitution_severity - cold.B0_condition_only
    cold["delta_b175_vs_b15"] = cold.B175_absolute_aa_identity - cold.B15_substitution_severity
    cold["delta_b2_vs_b175"] = cold.B2_position_context - cold.B175_absolute_aa_identity
    cold["delta_b2_vs_b15_total"] = cold.B2_position_context - cold.B15_substitution_severity
    cold.to_csv(OUT / "dms_b15_cold_contrasts.csv", index=False)
    summary = cold.groupby("challenge", observed=True).agg(
        folds=("repeat", "size"), b0_mae=("B0_condition_only", "mean"), b15_mae=("B15_substitution_severity", "mean"),
        b175_mae=("B175_absolute_aa_identity", "mean"), b2_mae=("B2_position_context", "mean"),
        delta_b15_vs_b0=("delta_b15_vs_b0", "mean"), delta_b175_vs_b15=("delta_b175_vs_b15", "mean"),
        delta_b2_vs_b175=("delta_b2_vs_b175", "mean"), delta_b2_vs_b15_total=("delta_b2_vs_b15_total", "mean")
    ).reset_index()
    summary.to_csv(OUT / "dms_b15_challenge_summary.csv", index=False)
    return cold, summary


def permuted_context(train, rng):
    positions = np.array(sorted(train.residue.unique()))
    permuted = rng.permutation(positions)
    mapping = dict(zip(positions, permuted))
    return train.residue.map(mapping).to_numpy(float)


def conditional_null(frame, observed, permutations, jobs=1):
    split_list = [s for s in outer_splits(frame) if s[0] == "random_variant_fixed_test"]
    observed_primary = float(observed.loc[observed.challenge.eq("random_variant_fixed_test"), "delta_b2_vs_b175"].mean())
    prepared = []
    for challenge, repeat, test_idx, cold_mask in split_list:
        test = frame.loc[test_idx].copy()
        train = frame.loc[~cold_mask].drop(index=test_idx, errors="ignore").copy()
        alpha_ref = tune(train, "B175_absolute_aa_identity")
        pred_ref = ridge(train, test, alpha_ref, "B175_absolute_aa_identity")
        prepared.append((repeat, train, test, wmae(test.y.to_numpy(), pred_ref, test.weight.to_numpy())))
    def one_draw(draw):
        fold_delta = []
        for repeat, train, test, ref_mae in prepared:
            rng = np.random.default_rng(SEED + draw * 1009 + repeat)
            ctx = permuted_context(train, rng)
            alpha_null = tune(train, "B2_position_context", ctx)
            pred_null = ridge(train, test, alpha_null, "B2_position_context", ctx, test.residue.to_numpy(float))
            fold_delta.append(wmae(test.y.to_numpy(), pred_null, test.weight.to_numpy()) - ref_mae)
        return {"draw": draw, "delta_b2null_vs_b175": float(np.mean(fold_delta))}
    draws = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(delayed(one_draw)(draw) for draw in range(1, permutations + 1))
    draws = pd.DataFrame(draws)
    draws.to_csv(OUT / "dms_position_conditional_null_draws.csv", index=False)
    values = draws.delta_b2null_vs_b175.to_numpy()
    lower_tail = float((np.sum(values <= observed_primary) + 1) / (len(values) + 1))
    upper_tail = float((np.sum(values >= observed_primary) + 1) / (len(values) + 1))
    result = {
        "primary_challenge": "random_variant_fixed_test",
        "contrast": "cold MAE(B2 position context) - cold MAE(B1.75 absolute-AA identity)",
        "observed_delta": observed_primary, "permutations": permutations,
        "null_mean_C_pos": float(values.mean()), "null_sd": float(values.std(ddof=1)),
        "null_q025": float(np.quantile(values, .025)), "null_q50": float(np.quantile(values, .5)), "null_q975": float(np.quantile(values, .975)),
        "observed_percentile": lower_tail, "lower_tail_probability": lower_tail, "upper_tail_probability": upper_tail,
        "decision_rule": "position increment if observed delta is below null distribution; report lower-tail probability",
    }
    (OUT / "dms_position_conditional_null_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--skip-null", action="store_true")
    parser.add_argument("--reuse-observed", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    contrast_path = OUT / "dms_b15_cold_contrasts.csv"
    summary_path = OUT / "dms_b15_challenge_summary.csv"
    if args.reuse_observed and contrast_path.exists() and summary_path.exists():
        observed, summary = pd.read_csv(contrast_path), pd.read_csv(summary_path)
    else:
        observed, summary = observed_ablation(frame)
    null = None if args.skip_null else conditional_null(frame, observed, args.permutations, args.jobs)
    meta = {
        "status": "COMPLETED" if null is not None or args.skip_null else "PARTIAL",
        "labels_sha256": sha(LABELS), "assignments_sha256": sha(ASSIGN), "seed": SEED,
        "feature_hierarchy": ["B0", "B1.5", "B1.75", "B2"],
        "identifiability_correction": "B1.75 inserted because B2-B1.5 mixes absolute amino-acid identity with position; C_pos calibrates B2-B1.75 only.",
        "b2_implementation_scope": "B1.5 + absolute WT/Mut identity + residue-position polynomial; no local sequence window, PLM, PCA, or 3D structure.",
        "prior_outputs_overwritten": False,
        "conditional_null": null,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    if null:
        print(json.dumps(null, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
