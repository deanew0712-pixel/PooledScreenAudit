"""Validated command-line interface around the frozen matrix analysis kernels."""
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from . import __version__

HERE = Path(__file__).resolve().parent


def kernel():
    spec = importlib.util.spec_from_file_location("psa_frozen_audit", HERE / "vendor/audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(df, command, controls="T_", targets="R_"):
    required = ["residue"]
    if command == "screen":
        required += ["variant", "wildtype", "mutant", "condition", "y", "weight"]
    missing = set(required) - set(df)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    tc = [c for c in df if c.startswith(controls)]
    rc = [c for c in df if c.startswith(targets)]
    if not tc:
        raise ValueError("No control columns found. Control association is not assessable without controls.")
    if not np.isfinite(pd.to_numeric(df.residue, errors="coerce")).all():
        raise ValueError("Residue must be numeric and finite.")
    df["residue"] = pd.to_numeric(df.residue, errors="raise")
    if df.residue.nunique() < 10:
        raise ValueError("At least ten distinct positions are required for this validation design.")
    for c in tc + (rc if command == "eta" else []):
        df[c] = pd.to_numeric(df[c], errors="raise")
        if np.isinf(df[c]).any():
            raise ValueError(f"Infinite values in {c}; use missing values instead.")
    if not any(df[c].notna().any() for c in tc):
        raise ValueError("All control entries are missing.")
    if command != "screen" and df.residue.duplicated().any():
        raise ValueError("Site input must contain exactly one row per residue.")
    if command == "screen":
        if df[required].isna().any().any():
            raise ValueError("Screen identifiers, response and weights cannot be missing.")
        if df.duplicated(["variant", "condition"]).any():
            raise ValueError("Duplicate variant-condition edges; aggregate deliberately upstream.")
        if (df.groupby("variant")[["residue", "wildtype", "mutant"]].nunique() > 1).any().any():
            raise ValueError("Each variant must map to one residue and amino-acid substitution.")
        aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(df.wildtype).issubset(aa) or not set(df.mutant).issubset(aa):
            raise ValueError("Use canonical one-letter amino-acid codes.")
        for c in ["y", "weight"]:
            df[c] = pd.to_numeric(df[c], errors="raise")
            if not np.isfinite(df[c]).all():
                raise ValueError(f"{c} must be finite.")
        if (df.weight <= 0).any():
            raise ValueError("Weights must be positive.")
        if (df.groupby("residue")[tc].nunique(dropna=False) > 1).any().any():
            raise ValueError("Controls must be consistent across edges at each residue.")
        if df.variant.nunique() < 10 or (df.groupby("condition").residue.nunique() < 10).any():
            raise ValueError("Each condition must span at least ten positions.")
        if df.y.nunique() < 2:
            raise ValueError("Constant response cannot support a predictive audit.")
    if command == "eta":
        if not rc:
            raise ValueError("No target columns found.")
        if any(df[c].notna().sum() < 5 or df[c].nunique() < 2 for c in rc):
            # Constant targets are deliberately handled by the frozen scoring kernel.
            if not any(df[c].notna().sum() >= 5 and df[c].nunique() >= 2 for c in rc):
                raise ValueError("No nonconstant target with enough observed values.")
        for width in [15, 20, 30]:
            if ((df.residue - df.residue.min()) // width).nunique() < 5:
                raise ValueError("Eta requires at least five coordinate groups at each of 15/20/30 residues.")
    return tc, rc


def partition_audit(df, tc, out, core, column, width, draws, seed):
    r = df.residue.to_numpy()
    rows = []
    rng = np.random.default_rng(seed)

    def record(label, g, draw):
        one = pd.get_dummies(g).to_numpy(float)
        d = core.directions(r, one)
        rows.append(dict(definition=label, replicate=draw, groups=len(np.unique(g)),
                         centered_rank=d["rank"] - 1,
                         **{f"shared_v{i+1}":float(v) for i, v in enumerate(d["shared"])}))
    groups = ((r-r.min()) // width).astype(int)
    record("analyst_bins", groups, 0)
    for j in range(draws):
        record("random_bins_same_sizes", rng.permutation(groups), j)
    if column:
        if column not in df or df[column].isna().any():
            raise ValueError("Partition column is absent or contains missing labels.")
        groups = df[column].astype(str).to_numpy()
        record("supplied_partition", groups, 0)
        for j in range(draws):
            record("random_partition_same_sizes", rng.permutation(groups), j)
    d = core.directions(r, df[tc].to_numpy(float))
    rows.append(dict(definition="T", replicate=0, groups=0, centered_rank=d["rank"]-1,
                     **{f"shared_v{i+1}":float(v) for i,v in enumerate(d["shared"])}))
    pd.DataFrame(rows).to_csv(out / "partition_geometry.csv", index=False)


def report(out, command):
    lines = ["# PooledScreenAudit report", "", f"Command: {command}", "",
             "Positive gain means lower MAE. Splits are not independent experiments.",
             "Control association and representation overlap do not establish causality.",
             "No automatic biological, clinical or separability classification is produced.", ""]
    if (out / "screen_summary.csv").exists():
        frame = pd.read_csv(out / "screen_summary.csv")
        for _, row in frame[frame.condition == "ALL"].iterrows():
            lines.append(f"- {row.challenge}: P gain {row.P_gain:.6g}; P after T {row.P_given_T:.6g}; "
                         f"relative gains {100*row.P_relative:.3f}% and {100*row.P_given_T_relative:.3f}%.")
        lines += ["", "Generic direction bases use outer-training positions and are held fixed during inner tuning.",
                  "The separate frozen EGFR route re-estimates directions inside inner training sets."]
    if (out / "eta_summary.json").exists():
        for row in json.loads((out / "eta_summary.json").read_text()):
            lines.append(f"- Block {row['width']}, {row.get('sensitivity','primary')}: equal-feature eta {row['equal_feature_eta']:.6g}.")
        lines += ["", "Eta is descriptive predictability under the declared feature weighting; no transported cutoff is applied."]
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("command", choices=["screen", "eta", "geometry", "egfr-geometry", "egfr", "fetch-src", "example"])
    ap.add_argument("--input", type=Path, help="CSV input for screen, eta or geometry")
    ap.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are refused")
    ap.add_argument("--controls", default="T_", help="Control column prefix (default T_)")
    ap.add_argument("--targets", default="R_", help="Eta target column prefix (default R_)")
    ap.add_argument("--simple", action="store_true", help="Screen baseline/P/T/TP only; omit direction subsets")
    ap.add_argument("--position-center", type=float, default=797, help="Declared polynomial origin, default 797 (historical EGFR/SRC basis)")
    ap.add_argument("--position-scale", type=float, default=100, help="Declared positive polynomial scale, default 100")
    ap.add_argument("--partition-column", help="Geometry only: supplied experimental partition labels")
    ap.add_argument("--bin-width", type=int, default=20, help="Geometry only: analyst bin width (default 20)")
    ap.add_argument("--draws", type=int, default=200, help="Geometry only: size-matched random assignments (default 200)")
    ap.add_argument("--seed", type=int, default=20260905, help="Geometry random seed; screen and eta retain frozen splits")
    ap.add_argument("--limit", type=int, help="EGFR smoke test outer split limit, never a full study result")
    args = ap.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("Output directory already exists; choose a new directory to preserve earlier runs.")
        if not np.isfinite([args.position_center,args.position_scale]).all() or args.position_scale <= 0:
            raise ValueError("Polynomial origin and scale must be finite, with positive scale.")
        if args.draws < 1 or args.bin_width < 1 or (args.limit is not None and args.limit < 1):
            raise ValueError("Draws, bin width and smoke-test limit must be positive.")
        if args.limit and args.command != "egfr":
            raise ValueError("--limit is available only for the frozen EGFR smoke test.")
        start = time.perf_counter()
        input_digest = None
        core = kernel()
        if args.command in ["screen", "eta", "geometry", "egfr-geometry"]:
            if args.input is None:
                raise ValueError("--input is required for matrix commands.")
            df = pd.read_csv(args.input)
            tc, rc = validate(df, args.command, args.controls, args.targets)
            input_digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
            if args.command == "egfr-geometry" and (len(df)!=268 or set(df.residue)!=set(range(712,980))):
                raise ValueError("egfr-geometry requires the historical 712–979 coordinate coverage.")
            if args.command == "egfr-geometry" and (args.position_center !=797 or args.position_scale !=100):
                raise ValueError("Historical EGFR geometry has a fixed polynomial basis.")
            if args.command == "geometry" and args.partition_column:
                if args.partition_column not in df or df[args.partition_column].isna().any():
                    raise ValueError("Partition labels must be present and nonmissing.")
            def basis(r):
                z = (np.asarray(r,float)-args.position_center)/args.position_scale
                return np.column_stack([z,z*z,z*z*z])
            core.pbasis = basis
            if np.linalg.matrix_rank(basis(np.unique(df.residue))-basis(np.unique(df.residue)).mean(0)) < 3:
                raise ValueError("Coordinate basis is rank deficient or ill-conditioned under this parameterization.")
        args.out.mkdir(parents=True)
        with threadpool_limits(limits=1):
            if args.command == "screen":
                core.screen(df, tc, args.out, not args.simple)
            elif args.command == "geometry":
                partition_audit(df, tc, args.out, core, args.partition_column, args.bin_width, args.draws, args.seed)
            elif args.command == "egfr-geometry":
                core.geometry(df, tc, args.out)
            elif args.command == "eta":
                x = np.column_stack([df[tc], core.pbasis(df.residue)])
                y = df[rc].to_numpy(float)
                rows = []
                for width in [15,20,30]:
                    groups = ((df.residue-df.residue.min())//width).to_numpy()
                    res, features = core.eta(x,y,groups)
                    res["width"] = width
                    features.insert(0,"feature",rc)
                    features.to_csv(args.out/f"eta_features_{width}.csv",index=False)
                    rows.append(res)
                (args.out/"eta_summary.json").write_text(json.dumps(rows,indent=2))
            elif args.command == "example":
                for f in (HERE/"examples").iterdir():
                    if f.is_file() and not f.name.startswith('.'):
                        shutil.copyfile(f,args.out/f.name)
            else:
                base = HERE/"vendor"
                cmd = [sys.executable]
                if args.command == "egfr":
                    base = base/"fully_nested_egfr"
                    cmd += [str(base/"run_paper_a_fully_nested_direction.py"), "--input-dir",str(base/"inputs"),"--out",str(args.out.resolve())]
                    if args.limit:
                        cmd += ["--limit",str(args.limit)]
                else:
                    cmd += [str(base/"prepare_src.py"),"--out",str(args.out.resolve())]
                subprocess.run(cmd,check=True)
        report(args.out,args.command)
        metadata = dict(version=__version__,python=platform.python_version(),numpy=np.__version__,
                        pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform(),
                        seconds=time.perf_counter()-start,input_sha256=input_digest,arguments=vars(args),
                        smoke_only=bool(args.limit),scientific_scope="Frozen kernels with validated interface; not raw assay reconstruction")
        (args.out/"RUN.json").write_text(json.dumps(metadata,default=str,indent=2))
        files = {str(f.relative_to(args.out)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in args.out.rglob('*') if f.is_file() and not f.name.startswith('.')}
        (args.out/"OUTPUT_SHA256.json").write_text(json.dumps(files,indent=2))
    except (ValueError,KeyError,np.linalg.LinAlgError,subprocess.CalledProcessError) as exc:
        ap.exit(2,f"Input or execution error: {exc}\n")

if __name__ == "__main__":
    main()
