"""
ABIDE application figures for:
"Covariance Kernels on Unordered Pair Spaces"

This script uses the processed ABIDE objects from the existing project
directory and produces publication-quality figures for the Mathematics paper.

Expected files in the directory supplied with --data-dir:

Required:
    aal_centroids.csv
    ZA_edge.npy
    ZG_edge.npy
    ZAG_edge.npy

Optional:
    region_labels.csv
    region_labels.txt
    aal_labels.csv

Outputs are written to the directory supplied with --output-dir.

The script is designed to support the Mathematics paper, so the focus is on:
    1. Brain geometry
    2. Relational spectra
    3. Cumulative trace and 95% truncation levels
    4. Effective dimension summaries
    5. Leading relational mode displayed on the brain
    6. Exact truncation error curves for the shrinkage operator
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------------------------------------------------
# User-editable paths
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ABIDE figures for unordered-pair covariance")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing aal_centroids.csv and edge-space .npy files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    return parser.parse_args()

ARGS = parse_args()
DATA_DIR = ARGS.data_dir.resolve()
OUT_DIR = ARGS.output_dir.resolve()
CENTROID_FILE = DATA_DIR / "aal_centroids.csv"
KERNEL_FILES = {
    "Anatomical": DATA_DIR / "ZA_edge.npy",
    "Functional": DATA_DIR / "ZG_edge.npy",
    "Combined": DATA_DIR / "ZAG_edge.npy",
}
TAU2 = ARGS.tau2
V_NOISE = ARGS.noise_variance

# Number of highlighted edges in the brain-network figure
N_TOP_EDGES = 30

# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_edge_index(n_nodes):
    """Return array of unordered node pairs (i, j), i<j."""
    pairs = []
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            pairs.append((i, j))
    return np.asarray(pairs, dtype=int)


def find_xyz_columns(df):
    cols_lower = {c.lower(): c for c in df.columns}
    candidates = [
        ("x", "y", "z"),
        ("cx", "cy", "cz"),
        ("coord_x", "coord_y", "coord_z"),
        ("centroid_x", "centroid_y", "centroid_z"),
        ("x_mm", "y_mm", "z_mm"),
    ]
    for triple in candidates:
        if all(t in cols_lower for t in triple):
            return [cols_lower[t] for t in triple]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 3:
        return numeric_cols[:3]

    raise ValueError("Could not identify x/y/z columns in aal_centroids.csv.")


def load_centroids(path):
    df = pd.read_csv(path)
    xyz_cols = find_xyz_columns(df)
    coords = df[xyz_cols].to_numpy(float)

    label_cols = [c for c in df.columns if c.lower() in {"label", "region", "name", "roi", "aal_label"}]
    if label_cols:
        labels = df[label_cols[0]].astype(str).tolist()
    else:
        labels = [f"ROI {i+1}" for i in range(coords.shape[0])]

    return coords, labels, df


def load_kernel_or_embedding(path):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path.name} must be a 2D array.")

    # If square and symmetric, treat as a kernel.
    if arr.shape[0] == arr.shape[1] and np.allclose(arr, arr.T, atol=1e-8):
        K = 0.5 * (arr + arr.T)
        source_type = "kernel"
    else:
        # Otherwise treat as an edge embedding / basis, and form K = Z Z^T.
        K = arr @ arr.T
        K = 0.5 * (K + K.T)
        source_type = "embedding"

    return K, arr, source_type


def safe_eigh(K, tol=1e-10):
    vals, vecs = np.linalg.eigh(0.5 * (K + K.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals[np.abs(vals) < tol * max(1.0, np.max(np.abs(vals)))] = 0.0
    vals[vals < 0] = 0.0
    return vals, vecs


def cumulative_trace(vals):
    s = np.sum(vals)
    if s <= 0:
        return np.zeros_like(vals)
    return np.cumsum(vals) / s


def effective_rank(vals):
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(np.sum(vals) / np.max(vals))


def entropy_rank(vals):
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    p = vals / np.sum(vals)
    return float(np.exp(-np.sum(p * np.log(p))))


def numerical_rank(vals, tol=1e-9):
    if vals.size == 0:
        return 0
    return int(np.sum(vals > tol * max(1.0, np.max(vals))))


def rank95(vals):
    c = cumulative_trace(vals)
    if c.size == 0:
        return 0
    return int(np.searchsorted(c, 0.95) + 1)


def truncation_curve(vals, tau2=1.0, v_noise=1.0):
    vals = np.asarray(vals)
    s = tau2 * vals / (tau2 * vals + v_noise)
    errors = []
    retained = []
    for r in range(1, len(vals)):
        errors.append(s[r])  # exact operator norm error from theorem
        retained.append(r)
    return np.asarray(retained), np.asarray(errors)


def summarize_kernel(K, name):
    vals, vecs = safe_eigh(K)
    return {
        "name": name,
        "K": K,
        "eigvals": vals,
        "eigvecs": vecs,
        "rank": numerical_rank(vals),
        "eff_rank": effective_rank(vals),
        "entropy_rank": entropy_rank(vals),
        "rank95": rank95(vals),
        "trace": float(np.sum(vals)),
    }


def line_widths(weights, min_w=0.5, max_w=3.0):
    weights = np.asarray(weights, float)
    if np.allclose(weights.max(), weights.min()):
        return np.full_like(weights, (min_w + max_w) / 2, dtype=float)
    scaled = (weights - weights.min()) / (weights.max() - weights.min())
    return min_w + (max_w - min_w) * scaled


# -------------------------------------------------------------------
# Figure 1: Node geometry
# -------------------------------------------------------------------

def plot_node_geometry(coords, out_file):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Simple coloring by hemisphere
    left = x < 0
    right = ~left

    views = [
        (x, y, "Axial view", "x", "y"),
        (y, z, "Sagittal view", "y", "z"),
        (x, z, "Coronal view", "x", "z"),
    ]

    for ax, (u, v, title, xlabel, ylabel) in zip(axes, views):
        ax.scatter(u[left], v[left], s=28, alpha=0.85, label="Left hemisphere")
        ax.scatter(u[right], v[right], s=28, alpha=0.85, marker="s", label="Right hemisphere")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25, linewidth=0.5)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7, label="Left hemisphere"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=7, label="Right hemisphere"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("AAL region geometry used for the ABIDE application", y=1.02)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Figure 2: Spectra
# -------------------------------------------------------------------

def plot_spectra(kernel_summaries, out_file):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for summary in kernel_summaries:
        vals = summary["eigvals"]
        pos = vals[vals > 0]
        idx = np.arange(1, len(pos) + 1)
        ax.semilogy(idx, pos, linewidth=1.8, label=summary["name"])
    ax.set_xlabel("Ordered eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Relational spectra for the ABIDE kernels")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Figure 3: Cumulative trace explained
# -------------------------------------------------------------------

def plot_cumulative_trace(kernel_summaries, out_file):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for summary in kernel_summaries:
        vals = summary["eigvals"]
        c = cumulative_trace(vals)
        idx = np.arange(1, len(c) + 1)
        ax.plot(idx, c, linewidth=1.8, label=summary["name"])
        r95 = summary["rank95"]
        if r95 > 0 and r95 <= len(c):
            ax.scatter([r95], [c[r95 - 1]], s=35)
            ax.text(r95, c[r95 - 1] + 0.02, f"{summary['name']}: {r95}", fontsize=9)

    ax.axhline(0.95, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Retained components")
    ax.set_ylabel("Cumulative trace explained")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative trace and 95% truncation levels")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Figure 4: Effective dimension summary
# -------------------------------------------------------------------

def plot_dimension_summary(kernel_summaries, out_file):
    labels = [s["name"] for s in kernel_summaries]
    ranks = [s["rank"] for s in kernel_summaries]
    effs = [s["eff_rank"] for s in kernel_summaries]
    ents = [s["entropy_rank"] for s in kernel_summaries]
    r95s = [s["rank95"] for s in kernel_summaries]

    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(x - 1.5*w, ranks, width=w, label="Numerical rank")
    ax.bar(x - 0.5*w, effs, width=w, label="Effective rank")
    ax.bar(x + 0.5*w, ents, width=w, label="Entropy rank")
    ax.bar(x + 1.5*w, r95s, width=w, label="95% trace rank")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dimension summary")
    ax.set_title("Intrinsic dimension of the ABIDE relational kernels")
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Figure 5: Leading relational mode on the brain
# -------------------------------------------------------------------

def plot_top_edges(ax, coords, pairs, scores, title, view="axial", n_top=30):
    idx = np.argsort(np.abs(scores))[::-1][:n_top]
    sel_pairs = pairs[idx]
    sel_scores = np.abs(scores[idx])

    if view == "axial":
        u = coords[:, 0]
        v = coords[:, 1]
        xlabel, ylabel = "x", "y"
    elif view == "sagittal":
        u = coords[:, 1]
        v = coords[:, 2]
        xlabel, ylabel = "y", "z"
    else:
        u = coords[:, 0]
        v = coords[:, 2]
        xlabel, ylabel = "x", "z"

    lw = line_widths(sel_scores)
    for (i, j), w in zip(sel_pairs, lw):
        ax.plot([u[i], u[j]], [v[i], v[j]], linewidth=w, alpha=0.75)
    ax.scatter(u, v, s=18, zorder=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20, linewidth=0.4)


def plot_leading_modes(coords, pairs, kernel_summaries, out_file, n_top=30):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, summary in zip(axes, kernel_summaries):
        scores = summary["eigvecs"][:, 0]
        plot_top_edges(
            ax,
            coords,
            pairs,
            scores,
            title=f"{summary['name']} kernel\nleading relational mode",
            view="axial",
            n_top=n_top,
        )
    fig.suptitle("Most prominent edges under the leading relational mode", y=1.03)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Figure 6: Exact truncation error curves
# -------------------------------------------------------------------

def plot_truncation_curves(kernel_summaries, out_file, tau2=1.0, v_noise=1.0):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for summary in kernel_summaries:
        r, err = truncation_curve(summary["eigvals"], tau2=tau2, v_noise=v_noise)
        ax.plot(r, err, linewidth=1.8, label=summary["name"])
    ax.set_xlabel("Retained rank")
    ax.set_ylabel("Exact shrinkage-operator error")
    ax.set_title("Exact truncation error for the relational estimator")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Save a manuscript-ready summary table
# -------------------------------------------------------------------

def save_summary(kernel_summaries, out_file):
    rows = []
    for s in kernel_summaries:
        rows.append({
            "kernel": s["name"],
            "trace": s["trace"],
            "numerical_rank": s["rank"],
            "effective_rank": s["eff_rank"],
            "entropy_rank": s["entropy_rank"],
            "rank_95pct_trace": s["rank95"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    return df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CENTROID_FILE.exists():
        raise FileNotFoundError(f"Missing centroid file: {CENTROID_FILE}")

    coords, labels, centroid_df = load_centroids(CENTROID_FILE)
    n_nodes = coords.shape[0]
    pairs = make_edge_index(n_nodes)
    q = pairs.shape[0]

    kernel_summaries = []

    print("\nLoading ABIDE kernels / embeddings")
    print("----------------------------------")
    for name, path in KERNEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        K, raw, source_type = load_kernel_or_embedding(path)

        if K.shape[0] != q or K.shape[1] != q:
            raise ValueError(
                f"{path.name} produced a {K.shape} matrix, but the ABIDE edge domain "
                f"for {n_nodes} nodes has q={q} edges."
            )

        summary = summarize_kernel(K, name)
        kernel_summaries.append(summary)

        print(f"{name:12s} | source={source_type:9s} | "
              f"rank={summary['rank']:4d} | "
              f"eff.rank={summary['eff_rank']:.2f} | "
              f"entropy.rank={summary['entropy_rank']:.2f} | "
              f"95% rank={summary['rank95']:4d}")

    save_summary(kernel_summaries, OUT_DIR / "abide_kernel_summary.csv")

    print("\nCreating figures")
    print("----------------")
    plot_node_geometry(coords, OUT_DIR / "figure_abide_brain_geometry.pdf")
    plot_spectra(kernel_summaries, OUT_DIR / "figure_abide_spectra.pdf")
    plot_cumulative_trace(kernel_summaries, OUT_DIR / "figure_abide_cumulative_trace.pdf")
    plot_dimension_summary(kernel_summaries, OUT_DIR / "figure_abide_dimension_summary.pdf")
    plot_leading_modes(coords, pairs, kernel_summaries, OUT_DIR / "figure_abide_leading_modes.pdf", n_top=N_TOP_EDGES)
    plot_truncation_curves(kernel_summaries, OUT_DIR / "figure_abide_truncation_curves.pdf", tau2=TAU2, v_noise=V_NOISE)

    print("\nFinished. Files written to:")
    print(OUT_DIR)

    print("\nSuggested figures for the Mathematics paper:")
    print("1. figure_abide_brain_geometry.pdf")
    print("2. figure_abide_spectra.pdf")
    print("3. figure_abide_cumulative_trace.pdf")
    print("4. figure_abide_dimension_summary.pdf")
    print("5. figure_abide_leading_modes.pdf")
    print("6. figure_abide_truncation_curves.pdf")


if __name__ == "__main__":
    main()
