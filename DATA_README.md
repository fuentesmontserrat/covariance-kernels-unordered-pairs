# Data inputs

The public ABIDE I source is:
https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html

The manuscript uses quality-checked ABIDE I preprocessed regional time-series derivatives and a 116-region AAL parcellation. The source data are third-party data and are not redistributed by this repository.

For the theorem-driven application scripts, generate or place the following derived files in a local input directory:

- `aal_centroids.csv`: one row per AAL region with x/y/z coordinates.
- `ZA_edge.npy`: anatomical edge-space feature matrix.
- `ZG_edge.npy`: diagnosis-blind functional edge-space feature matrix.
- `ZAG_edge.npy`: combined anatomical-functional feature matrix.

The code is intentionally location-agnostic: pass the input directory with `--data-dir` rather than editing paths inside a script.
