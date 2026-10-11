# EGFR site matrix

`egfr_sites.csv` is a derived numerical feature matrix, not patient data or raw sequencing reads. It combines the project's continuous-structure atlas (22 numeric features from PDB 4ZAU and 8D76) and DMSO site-QC atlas (21 numeric features derived from GSE305057), over 268 residues. Original analysis provenance is retained in the historical companion DOI 10.5281/zenodo.22306788 and its predecessor. Source publication: Wang et al., NPJ Precision Oncology 2025, DOI 10.1038/s41698-025-01086-2. Source data remain subject to their original terms.

Construction from the historical project's `outputs/freeze_amendment_v3_2026-08-26/outcome_blind_atlases/`:

1. Read `continuous_structure_feature_atlas.csv`; retain residue as key and prefix all other column names with `R_`.
2. Read `dmso_site_technical_qc_atlas.csv`; remove `site` and `wildtype_codon`, retain residue as key and prefix other names with `T_`.
3. Merge by residue, preserving structural-atlas row order. Do not fill missing values before training folds are constructed.

The matrix is included for execution without reconstructing the historical project. The enclosing archive's SHA256 manifest records its bytes. This convenience input does not make the package a complete raw-count-to-structure pipeline. `geometry` uses the declared source boundary at residue 846; it is not a generic library-tile inference procedure.
