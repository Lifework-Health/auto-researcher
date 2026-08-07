# Iris dataset attribution

- Dataset: Iris
- Canonical source: UCI Machine Learning Repository, dataset 53
- DOI: https://doi.org/10.24432/C56C76
- Creator and original reference: R. A. Fisher, “The use of multiple measurements in taxonomic problems” (1936)
- UCI file: `bezdekIris.data`, retrieved 2026-08-06 from `https://archive.ics.uci.edu/static/public/53/iris.zip`
- UCI archive SHA-256 at retrieval: `d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9`
- Vendored file SHA-256: `0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc`

UCI distributes this dataset under the [Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/). Redistribution and adaptation are permitted with attribution.

The vendored data file is byte-for-byte identical to UCI’s `bezdekIris.data`. No observations, numeric values, or labels were transformed. `folds-v1.json` is an Auto Researcher addition: within each canonical 50-row class block, the zero-based within-class row offset is assigned modulo five, producing five validation folds with ten observations per species.
