# Input provenance and third party terms

The scientific Python kernels are copied from the authors' matrix companion. Their checksums and relative source paths are in KERNEL_PROVENANCE.json. Code retains the accompanying MIT license. No upstream dataset is relicensed as MIT.

The bundled site matrix and frozen EGFR edge matrix are derived research inputs from GSE305057 and doi:10.1038/s41698-025-01086-2. See examples/PROVENANCE.md, the frozen input files, and the historical analysis companion doi:10.5281/zenodo.22707659. They do not reconstruct the source authors' raw-count processing. Retain source attribution and consult source terms for reuse.

SRC counts originate from https://github.com/eahler/2023_src_inhib at commit 956034184cbce1c65d34770557661b9a5905b2a3 (Chakraborty et al., Cell Chemical Biology, doi:10.1016/j.chembiol.2023.08.005). The five raw count files are deliberately not bundled. The fetch command retrieves and verifies them from the source. No explicit upstream repository license was established; this software does not confer redistribution rights.
