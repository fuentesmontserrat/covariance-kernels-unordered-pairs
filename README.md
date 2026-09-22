# Covariance Kernels on Unordered Pair Spaces — Reproducibility Code

This repository contains platform-independent Python code accompanying the manuscript
**“Covariance Kernels on Unordered Pair Spaces: Theory and Applications to Network-Valued Data.”**
https://doi.org/10.20944/preprints202608.1312.v1

No script contains an author-specific absolute path. Input and output locations are supplied at the command line.

## Data

The neuroimaging application uses ABIDE I, available through the 1000 Functional Connectomes Project / INDI:
https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html

ABIDE data are not redistributed here. Access is governed by the source repository. The application scripts use derived, de-identified numerical inputs created from the ABIDE workflow described in the manuscript. Place these files in a directory of your choice:

- `aal_centroids.csv`
- `ZA_edge.npy`
- `ZG_edge.npy`
- `ZAG_edge.npy`

The first file contains the 116 AAL centroid coordinates. The three `.npy` files contain edge-space feature/basis matrices for the anatomical, functional, and combined geometries. The scripts reconstruct all spectral summaries from these matrices.

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Simulations

```bash
python simulations.py --output-dir results/simulations
```

The manuscript seed is `20260814`. Full replication counts are 5,000 for the risk experiment, 1,000 for truncation, and 500 for perturbation. Use `--quick` for a short test run.

## ABIDE theorem-linked application

```bash
python abide_application_theorem_results.py \
  --data-dir /path/to/derived_data \
  --output-dir results/application
```

## ABIDE figures

```bash
python abide_application_figures.py \
  --data-dir /path/to/derived_data \
  --output-dir results/figures
```

## Reproducibility notes

- Numerical rank uses the relative threshold `lambda_j > 1e-10 * lambda_1` in the theorem-linked application script.
- The illustrative shrinkage scale used in the manuscript application is `tau^2 = v = 1`; it can be changed through command-line arguments in the theorem-results script.
- The ABIDE functional geometry is diagnosis-blind and is treated as fixed in this theorem-driven application. It should not be interpreted as an empirical estimate of the full edge covariance.
