# Publication checklist

The engineering candidate passed the 12-job OS/Python matrix and Linux container job on 11 October 2026. See CI_VALIDATION.json for the exact tested commit. Stable-release commits are tested again before tagging. PyPI, a new Zenodo DOI, Software Heritage and bio.tools remain separate publication steps; do not infer their completion from the GitHub release.

1. Approved repository: deanew0712-pixel/PooledScreenAudit. Include source, tests, CI, license, README and attribution. Do not upload manuscripts, personal correspondence or unlicensed raw counts.
2. Push and inspect all 12 Python/OS jobs and the Linux container job. Fix failures; record run URLs and exact commits. Do not replace a failed platform with an unsupported claim.
3. Build sdist and wheel, run `python -m twine check dist/*`, install the wheel in a fresh environment and run the documented checks.
4. Author creates a PyPI/TestPyPI project and approves publication. Prefer trusted publishing. Never put API tokens into this directory. Confirm package-name availability and maintainership before upload.
5. After approval, tag the final release and archive its exact bytes on Zenodo. The current 0.4.0 DOI does not identify this candidate.
6. Request Software Heritage archival of the public Git repository and copy the returned SWHID; do not fabricate one. Register bio.tools with the confirmed repository, release DOI and RRID SCR_029094.
7. Optionally build a Code Ocean capsule, document dependencies, input terms and a small offline reproduction. A Dockerfile alone is not an executed capsule.
8. Update the manuscript only with completed public releases and actual platform test results.

Reference documentation: https://docs.github.com/en/actions/tutorials/build-and-test-code/python ; https://packaging.python.org/en/latest/tutorials/packaging-projects/ ; https://docs.pypi.org/trusted-publishers/ ; https://archive.softwareheritage.org/save/ ; https://bio.tools/
