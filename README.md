# PooledScreenAudit

PooledScreenAudit evaluates position-based prediction and control association in pooled variant-effect matrices. It provides matched predictive models, control adjustment, direction attribution and rank-aware geometry. This 0.4.1rc1 engineering candidate retains the 0.4.0 numerical kernels and adds tests, CI configuration and a container recipe.

This candidate is not published. The stable 0.4.0 release is https://doi.org/10.5281/zenodo.22846913. Supporting research analyses are archived separately at https://doi.org/10.5281/zenodo.22707659. RRID:SCR_029094. Neither DOI identifies this unreleased candidate. See RELEASE_CHECKLIST.md for publication steps and actual validation records for tested platforms.

## Install and quick start

Use Python 3.9–3.12 in a fresh virtual environment; the release test used Python 3.9.6. Python and a CPU are sufficient; no GPU or credentials are required. Installation uses standard Python dependencies. Version bounds are in pyproject.toml; tested environments and timings are in RELEASE_VALIDATION.json, not claims of testing every allowed interpreter or platform.

```sh
python -m venv env
env/bin/python -m pip install /absolute/path/to/pooledscreenaudit
env/bin/pooledscreenaudit --version
env/bin/pooledscreenaudit example --out example_input
env/bin/pooledscreenaudit eta --input example_input/egfr_sites.csv --out example_eta
env/bin/pooledscreenaudit egfr-geometry --input example_input/egfr_sites.csv --out example_geometry
```

On Windows use `env\Scripts\python.exe` and `env\Scripts\pooledscreenaudit.exe`. Windows has not been tested in this release. Outputs must not already exist, preventing accidental replacement of previous analyses.

## Commands and parameters

Run `pooledscreenaudit --help` for complete options. All commands require `--out` (new output directory).

| Command | Purpose | Input and key options |
|---|---|---|
| example | Copy offline EGFR site example and provenance | No input argument |
| eta | Predict structure features from controls and cubic position | `--input` site CSV; `--controls T_`; `--targets R_` |
| geometry | Compare control and partition overlap with cubic position | `--input` site CSV; optional `--partition-column`; `--bin-width 20`; `--draws 200`; `--seed 20260905` |
| egfr-geometry | Reproduce frozen source-subamplicon/cutpoint/random-bin comparison | EGFR site CSV spanning 712–979; fixed historical boundary 846 |
| screen | Baseline/P/T/TP and optional direction-subset prediction | `--input` edge CSV; `--simple` omits direction subset fits |
| egfr | Reproduce frozen fully inner-nested EGFR direction analysis | Bundled derived edges/splits; optional `--limit 1` is a smoke test only |
| fetch-src | Download and hash-check pinned source counts, then derive SRC edges | Network required; source provenance recorded |

`--position-center 797` and `--position-scale 100` declare the cubic basis for matrix commands. These defaults preserve historical EGFR and SRC analyses; choose a scientifically appropriate parameterization before analyzing a new assay. Scaling/origin changes can affect ridge regularization and direction results, so they are not declared invariant. The frozen `egfr` and `egfr-geometry` routes retain their original parameterization. The seed option controls only new `geometry` random assignments; screen/eta retain their documented fixed split designs. `--limit` affects only the EGFR reproduction command.

## Input contract

Site tables: one row per finite numeric `residue`, control columns beginning `T_`, and, for eta, target columns beginning `R_`. Missing feature values are allowed, infinite values are not. Eta requires at least five blocks at each of widths 15, 20 and 30. Constant targets are excluded by the scoring kernel; all-constant target sets are rejected. Partially missing targets are scored only where observed. Inputs without controls are rejected rather than assigned a control-associated interpretation.

Edge tables: `variant,residue,wildtype,mutant,condition,y,weight` plus `T_` columns. Each variant has one position/substitution; there must be one row per variant-condition edge. Use canonical one-letter amino acids, finite responses and positive finite weights. Site-level controls must be identical, including missingness, across edges at a residue. At least ten positions per condition are required. This is a numerical guard, not a guarantee of adequate statistical power. Build responses, weights and control atlases upstream and document their relationship. Do not include patient identifiers.

For a new assay:

```sh
pooledscreenaudit screen --input assay_edges.csv --out assay_audit --simple
pooledscreenaudit geometry --input assay_sites.csv --out assay_geometry --partition-column sequencing_partition
```

The supplied partition label establishes provenance only if the user has source documentation. Analyst bins and sequencing partitions are not library-synthesis tiles. Null assignments preserve group sizes and centered indicator rank, not experimental causal structure.

## Reproduce the worked examples

```sh
pooledscreenaudit egfr --out egfr_nested
pooledscreenaudit fetch-src --out src_input
pooledscreenaudit screen --input src_input/src_edges.csv --out src_audit --simple
```

SRC raw files are downloaded from the pinned commit in `vendor/expected_SRC_PROVENANCE.json` and SHA256-checked. They are not redistributed or relicensed. SRC y is mean-replicate log2 endpoint-C/input frequency, not the published activity score and not automatically resistance. Controls are input-count summaries, not DMSO. The example tests execution in another assay; it is not independent same-task EGFR validation.

## Validation scope and interpretation

Generic screen: five repeated variant-cold 80/20 splits and five position-cold groups. All edges for a held-out variant remain in the test set. Three grouped inner folds select ridge alpha from 0.01, 0.1, 1, 10, 100. Predictive imputation/scaling are learned inside each training fit. Directions are estimated from outer-training positions and held fixed during inner tuning; **this direction procedure is not fully nested**. Baseline is amino-acid identity with condition indicators and interactions. The optional full mode fits all subsets and computes exact three-player Shapley allocations in this specified model basis.

Frozen EGFR: ten variant-cold and five position-cold evaluations, with the historical substitution-plus-identity baseline and selected test edges. Directions and predictive preprocessing are inner-training fitted, conditional on frozen upstream labels, controls and reliability weights. The EGFR assay construction itself is not nested. Do not pool these results with generic screen results as if baselines and sampling were identical.

Eta: five block-grouped outer folds and three inner folds; target and predictor preprocessing is training-fitted. The reported score is the square root of the nonnegative part of mean feature-wise out-of-fold R², after error/deviation sums are normalized by outer-training feature variance. Negative R² contributes to the mean. It measures predictability, not empirical identifiability. The wrapper reports the three primary block widths; historical solvent-accessibility sensitivities remain in Source Data 4.

Positive MAE gain means improvement. Negative gains are retained. Paired split distributions are descriptive, not independent biological replicate confidence intervals. Small adjusted gain is insufficient evidence without a meaningful unadjusted signal. No universal cutoff, causal technical mechanism, disease prediction or treatment recommendation is generated. Missing control metadata and shared outcome denominators limit interpretation.

## Outputs and provenance

`REPORT.md` summarizes available metrics and scope; `RUN.json` records versions, invocation, input SHA256 and wall time; `OUTPUT_SHA256.json` hashes outputs. Scientific CSV/JSON files retain legacy names for comparison with historical results. `KERNEL_PROVENANCE.json` verifies that the copied kernels and frozen EGFR inputs match their source files. Geometry commands export spectra in CSV; interpreting experimental origin requires the source metadata. `--limit` runs are flagged as smoke-only in RUN.json.

## Run the offline checks

From the extracted source directory after installation:

```sh
python -m unittest discover -s tests -v
python tests/numerical_fixture.py
```

NumPy is restricted to the 1.26 series: NumPy 2.0.2 triggered a floating-point warning in the frozen EGFR solver self-check on the tested macOS system. See RELEASE_VALIDATION.json and the dependency lock file in the release for the tested combination. Availability of a wheel is not a claim that every operating system was tested.

## License and support

Code retains MIT terms (LICENSE). Third-party experimental inputs retain their source terms; see THIRD_PARTY_NOTICES.md. This package does not grant rights to raw upstream data. Report software issues at https://github.com/deanew0712-pixel/PooledScreenAudit/issues. A public repository does not imply a guaranteed response time or maintenance commitment.
