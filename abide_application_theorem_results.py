"""
Theorem-driven ABIDE application analysis
=========================================

For the Mathematics paper:
"Covariance Kernels on Unordered Pair Spaces:
 Theory and Applications to Network-Valued Data"

This script uses derived ABIDE inputs supplied through --data-dir.

It complements abide_application_figures.py.  The first script makes descriptive
figures.  This script produces theorem-linked application RESULTS and TABLES.

Required files
--------------
data/processed/aal_centroids.csv
data/processed/ZA_edge.npy
data/processed/ZG_edge.npy
data/processed/ZAG_edge.npy

Optional
--------
outputs/v3/v3_asd_edge_results.csv
data/processed/v3_asd_edge_results.csv

Outputs
-------
results/math_application_theorem_results/

    table1_spectral_complexity.csv
    table2_inferential_truncation.csv
    table3_anatomical_loopfree_restriction.csv
    table4_anatomical_perturbation.csv
    table5_effect_alignment.csv            (only if posterior results found)

    figure1_spectra_and_trace.pdf
    figure2_inferential_truncation.pdf
    figure3_anatomical_loopfree.pdf
    figure4_anatomical_perturbation.pdf

    application_summary.txt

Dependencies
------------
numpy pandas matplotlib scipy

Install once if needed:
    python -m pip install numpy pandas matplotlib scipy

Run
---
python abide_application_theorem_results.py --data-dir /path/to/derived_data --output-dir results/application
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh

# ---------------------------------------------------------------------
# Paths/settings
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Theorem-linked ABIDE relational-covariance analysis")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing aal_centroids.csv and ZA_edge/ZG_edge/ZAG_edge .npy files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/application"))
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()

ARGS = parse_args()
P = ARGS.data_dir.resolve()
OUT = ARGS.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
CENTROIDS = P / "aal_centroids.csv"
FEATURE_FILES = {
    "Anatomical": P / "ZA_edge.npy",
    "Functional": P / "ZG_edge.npy",
    "Combined": P / "ZAG_edge.npy",
}
POSTERIOR_CANDIDATES = [P / "v3_asd_edge_results.csv"]
RHO_A = 31.0
TAU2 = ARGS.tau2
V = ARGS.noise_variance
TRACE_TARGETS = (0.90, 0.95, 0.99)
ERROR_TARGETS = (0.10, 0.05, 0.01)

# Perturb anatomical centroids by these SDs, in atlas-coordinate units.
PERTURB_SD = (0.5, 1.0, 2.0, 5.0)
PERTURB_REPS = 100
SEED = ARGS.seed

# Leading loop-free eigenvalues checked for the anatomy theorem illustration.
N_INTERLACE = 150

plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")
    return path


def find_xyz(df):
    low = {c.lower(): c for c in df.columns}
    candidates = [
        ("x","y","z"),
        ("cx","cy","cz"),
        ("centroid_x","centroid_y","centroid_z"),
        ("coord_x","coord_y","coord_z"),
        ("x_mm","y_mm","z_mm"),
    ]
    for triple in candidates:
        if all(v in low for v in triple):
            return [low[v] for v in triple]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) >= 3:
        return numeric[:3]
    raise ValueError("Could not identify x/y/z columns in aal_centroids.csv")


def eig_from_edge_features(Z):
    """
    Existing processed objects are edge feature matrices Z.
    The induced kernel is K = Z Z^T.
    Nonzero eigenvalues of K are singular-values(Z)^2.
    """
    Z = np.asarray(Z, float)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    vals = np.maximum(s*s, 0.0)
    return vals, U


def cumulative(vals):
    vals = np.asarray(vals, float)
    vals = vals[vals > 0]
    return np.cumsum(vals) / np.sum(vals)


def rank_for_trace(vals, p):
    c = cumulative(vals)
    return int(np.searchsorted(c, p) + 1)


def numerical_rank(vals, tol=1e-10):
    vals = np.asarray(vals)
    return int(np.sum(vals > tol * vals.max()))


def effective_rank(vals):
    vals = np.asarray(vals)
    vals = vals[vals > 0]
    return float(vals.sum() / vals.max())


def entropy_rank(vals):
    vals = np.asarray(vals)
    vals = vals[vals > 0]
    p = vals / vals.sum()
    return float(np.exp(-np.sum(p*np.log(p))))


def shrinkage(vals):
    vals = np.asarray(vals, float)
    return TAU2*vals / (TAU2*vals + V)


def rank_for_error(vals, target):
    """
    Exact theorem:
    ||S(K)-S(K_r)||_2 =
    tau^2 lambda_{r+1}/(tau^2 lambda_{r+1}+v).
    """
    s = shrinkage(vals)
    for r in range(1, len(s)):
        if s[r] <= target:
            return r
    return len(s)


def gaussian_node_kernel(coords, rho):
    sq = np.sum(coords*coords, axis=1)
    d2 = sq[:,None] + sq[None,:] - 2*coords@coords.T
    d2 = np.maximum(d2, 0.0)
    K = np.exp(-d2/(2*rho*rho))
    K = 0.5*(K+K.T)
    np.fill_diagonal(K, 1.0)
    return K


def make_pairs(R):
    i,j = np.triu_indices(R, 1)
    return np.column_stack((i,j))


def pair_operator(Kv):
    """
    Matrix-free loop-free pair kernel.
    """
    R = Kv.shape[0]
    pairs = make_pairs(R)
    ii,jj = pairs[:,0], pairs[:,1]
    q = len(pairs)

    def mv(x):
        X = np.zeros((R,R))
        X[ii,jj] = x
        X[jj,ii] = x
        Y = 0.5*(Kv @ X @ Kv)
        return Y[ii,jj]

    return LinearOperator((q,q), matvec=mv, rmatvec=mv, dtype=float)


def full_product_spectrum(node_eigs):
    lam = np.sort(np.maximum(node_eigs,0))[::-1]
    vals = []
    for j in range(len(lam)):
        vals.append(lam[j]**2)
        for k in range(j+1,len(lam)):
            vals.append(lam[j]*lam[k])
    return np.sort(np.asarray(vals))[::-1]


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)

cent = pd.read_csv(require(CENTROIDS))
xyz = find_xyz(cent)
coords = cent[xyz].to_numpy(float)
R = len(coords)
q = R*(R-1)//2
qfull = R*(R+1)//2

print("\nABIDE theorem-driven application")
print("--------------------------------")
print(f"Regions: {R}")
print(f"Loop-free unordered edges: {q}")
print(f"Full symmetric pair domain: {qfull}")

summaries = {}
rows = []

for name, path in FEATURE_FILES.items():
    Z = np.load(require(path)).astype(float)
    if Z.shape[0] != q:
        raise ValueError(f"{path.name}: expected {q} edge rows, found {Z.shape[0]}")
    vals, U = eig_from_edge_features(Z)
    summaries[name] = {"Z":Z, "vals":vals, "U":U}

    row = {
        "geometry": name,
        "edge_dimension": q,
        "basis_columns": Z.shape[1],
        "numerical_rank": numerical_rank(vals),
        "effective_rank": effective_rank(vals),
        "entropy_rank": entropy_rank(vals),
    }
    for p in TRACE_TARGETS:
        row[f"rank_{int(100*p)}pct_trace"] = rank_for_trace(vals,p)
        row[f"fraction_dimension_{int(100*p)}pct"] = rank_for_trace(vals,p)/q
    rows.append(row)

table1 = pd.DataFrame(rows)
table1.to_csv(OUT/"table1_spectral_complexity.csv", index=False)

print("\nTABLE 1: Spectral complexity")
print(table1.to_string(index=False))


# ---------------------------------------------------------------------
# Table 2: theorem-based inferential truncation
# ---------------------------------------------------------------------

rows = []
for name,s in summaries.items():
    vals = s["vals"]
    row = {"geometry":name, "tau2":TAU2, "noise_variance":V}
    for p in TRACE_TARGETS:
        r = rank_for_trace(vals,p)
        row[f"rank_{int(100*p)}pct_trace"] = r
        row[f"operator_error_at_{int(100*p)}pct_rank"] = (
            shrinkage(vals)[r] if r < len(vals) else 0.0
        )
    for eps in ERROR_TARGETS:
        row[f"rank_operator_error_below_{str(eps).replace('.','p')}"] = rank_for_error(vals,eps)
    rows.append(row)

table2 = pd.DataFrame(rows)
table2.to_csv(OUT/"table2_inferential_truncation.csv", index=False)

print("\nTABLE 2: Inferential truncation")
print(table2.to_string(index=False))


# ---------------------------------------------------------------------
# Table 3: loop-free theorem on the actual anatomical node geometry
# ---------------------------------------------------------------------

KA_node = gaussian_node_kernel(coords, RHO_A)
node_eigs = np.linalg.eigvalsh(KA_node)
mu = full_product_spectrum(node_eigs)

op = pair_operator(KA_node)
k = min(N_INTERLACE, q-2)
nu = eigsh(op, k=k, which="LA", return_eigenvectors=False, tol=1e-8)
nu = np.sort(np.maximum(nu,0))[::-1]

m = min(len(nu), len(mu)-R)
upper = np.maximum(nu[:m]-mu[:m],0)
lower = np.maximum(mu[R:R+m]-nu[:m],0)
max_violation = max(float(upper.max()), float(lower.max()))

trace_loss = float(np.sum(np.diag(KA_node)**2))
trace_full = float(mu.sum())
trace_loss_fraction = trace_loss/trace_full
bound = 4/(R+1)

table3 = pd.DataFrame([{
    "geometry":"Anatomical",
    "nodes":R,
    "full_symmetric_dimension":qfull,
    "loopfree_dimension":q,
    "leading_eigenvalues_checked":m,
    "trace_full":trace_full,
    "trace_loss_from_removing_self_pairs":trace_loss,
    "trace_loss_fraction":trace_loss_fraction,
    "generic_bound_4_over_Rplus1":bound,
    "max_leading_interlacing_violation":max_violation,
}])
table3.to_csv(OUT/"table3_anatomical_loopfree_restriction.csv", index=False)

print("\nTABLE 3: Loop-free restriction")
print(table3.to_string(index=False))


# ---------------------------------------------------------------------
# Table 4: perturb the real anatomical geometry
# ---------------------------------------------------------------------

rng = np.random.default_rng(SEED)
base_norm = float(np.linalg.norm(KA_node,2))
pert_rows = []

for sd in PERTURB_SD:
    for b in range(PERTURB_REPS):
        coords2 = coords + rng.normal(0,sd,size=coords.shape)
        K2 = gaussian_node_kernel(coords2,RHO_A)

        node_err = float(np.linalg.norm(KA_node-K2,2))
        bound_pair = 0.5*(base_norm + float(np.linalg.norm(K2,2)))*node_err

        # Compute pair-operator error by leading singular/eigen magnitude
        # via a matrix-free difference operator.
        op1 = pair_operator(KA_node)
        op2 = pair_operator(K2)
        diff = LinearOperator(
            op1.shape,
            matvec=lambda x, a=op1, b2=op2: a.matvec(x)-b2.matvec(x),
            rmatvec=lambda x, a=op1, b2=op2: a.matvec(x)-b2.matvec(x),
            dtype=float,
        )
        ev = eigsh(diff, k=1, which="LM", return_eigenvectors=False, tol=1e-6)
        pair_err = float(abs(ev[0]))

        pert_rows.append({
            "centroid_perturbation_sd":sd,
            "replicate":b+1,
            "node_operator_error":node_err,
            "pair_operator_error":pair_err,
            "pair_operator_bound":bound_pair,
            "actual_to_bound_ratio":pair_err/bound_pair if bound_pair>0 else np.nan,
        })

pert = pd.DataFrame(pert_rows)
table4 = pert.groupby("centroid_perturbation_sd",as_index=False).agg(
    node_operator_error_mean=("node_operator_error","mean"),
    pair_operator_error_mean=("pair_operator_error","mean"),
    pair_operator_error_sd=("pair_operator_error","std"),
    pair_operator_bound_mean=("pair_operator_bound","mean"),
    actual_to_bound_ratio_mean=("actual_to_bound_ratio","mean"),
    actual_to_bound_ratio_max=("actual_to_bound_ratio","max"),
)
table4.to_csv(OUT/"table4_anatomical_perturbation.csv", index=False)

print("\nTABLE 4: Anatomical geometry perturbation")
print(table4.to_string(index=False))


# ---------------------------------------------------------------------
# Optional Table 5: fitted-effect energy in relational eigenspaces
# ---------------------------------------------------------------------

posterior_file = next((p for p in POSTERIOR_CANDIDATES if p.exists()), None)

if posterior_file is not None:
    eff = pd.read_csv(posterior_file)
    effect_col = next(
        (c for c in ["posterior_mean","mean","asd_effect","effect"] if c in eff.columns),
        None
    )
    if effect_col is not None and len(eff)==q:
        beta = eff[effect_col].to_numpy(float)
        total_energy = float(beta@beta)
        align_rows = []

        for name,s in summaries.items():
            U = s["U"]
            scores = U.T @ beta
            energy = np.cumsum(scores*scores)
            for r in [10,25,50,100,195,360]:
                rr = min(r,U.shape[1])
                align_rows.append({
                    "geometry":name,
                    "retained_directions":rr,
                    "effect_energy_fraction":float(energy[rr-1]/total_energy),
                })

        table5 = pd.DataFrame(align_rows)
        table5.to_csv(OUT/"table5_effect_alignment.csv", index=False)
        print("\nTABLE 5: Effect alignment")
        print(table5.to_string(index=False))
    else:
        print("\nPosterior file found but no usable posterior_mean column or edge length mismatch.")
else:
    print("\nNo posterior edge-results file found. Table 5 skipped.")


# ---------------------------------------------------------------------
# Figure 1: spectra + cumulative trace
# ---------------------------------------------------------------------

fig,axes = plt.subplots(1,2,figsize=(11.2,4.5))
for name,s in summaries.items():
    vals=s["vals"]
    axes[0].semilogy(np.arange(1,len(vals)+1),vals,linewidth=1.7,label=name)
    axes[1].plot(np.arange(1,len(vals)+1),cumulative(vals),linewidth=1.7,label=name)

axes[0].set(
    xlabel="Ordered relational eigenvalue",
    ylabel="Eigenvalue (log scale)",
    title="A. ABIDE relational spectra"
)
axes[1].axhline(.95,linestyle="--",linewidth=1.1)
axes[1].set(
    xlabel="Retained directions",
    ylabel="Cumulative trace explained",
    title="B. Spectral compression"
)
axes[1].set_ylim(0,1.01)
for ax in axes:
    ax.grid(alpha=.22)
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/"figure1_spectra_and_trace.pdf",bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: exact inferential truncation error
# ---------------------------------------------------------------------

fig,ax=plt.subplots(figsize=(7.2,5))
for name,s in summaries.items():
    vals=s["vals"]
    sh=shrinkage(vals)
    r=np.arange(1,len(vals))
    exact=sh[1:]
    ax.semilogy(r,exact,linewidth=1.7,label=name)

for eps in ERROR_TARGETS:
    ax.axhline(eps,linestyle="--",linewidth=.9)

ax.set(
    xlabel="Retained rank",
    ylabel="Exact shrinkage-operator error",
    title="Inferential consequence of spectral truncation"
)
ax.grid(alpha=.22)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/"figure2_inferential_truncation.pdf",bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: loop-free interlacing in real anatomy
# ---------------------------------------------------------------------

fig,ax=plt.subplots(figsize=(7.2,5))
j=np.arange(1,m+1)
ax.semilogy(j,mu[:m],linewidth=1.5,label=r"Full spectrum $\mu_j$")
ax.semilogy(j,nu[:m],linewidth=1.5,label=r"Loop-free spectrum $\nu_j$")
ax.semilogy(j,mu[R:R+m],linewidth=1.5,label=r"Lower interlacing bound $\mu_{j+R}$")
ax.set(
    xlabel="Leading eigenvalue index",
    ylabel="Eigenvalue (log scale)",
    title="Loop-free interlacing in the ABIDE anatomical geometry"
)
ax.grid(alpha=.22)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/"figure3_anatomical_loopfree.pdf",bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Figure 4: perturbation theorem
# ---------------------------------------------------------------------

fig,ax=plt.subplots(figsize=(7.2,5))
ax.plot(
    table4["centroid_perturbation_sd"],
    table4["pair_operator_error_mean"],
    marker="o",
    linewidth=1.6,
    label="Observed pair-operator error"
)
ax.plot(
    table4["centroid_perturbation_sd"],
    table4["pair_operator_bound_mean"],
    marker="s",
    linestyle="--",
    linewidth=1.6,
    label="Theorem bound"
)
ax.set(
    xlabel="SD of centroid perturbation",
    ylabel="Operator norm",
    title="Propagation of anatomical-geometry perturbation"
)
ax.grid(alpha=.22)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/"figure4_anatomical_perturbation.pdf",bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# Short text summary for manuscript drafting
# ---------------------------------------------------------------------

with open(OUT/"application_summary.txt","w",encoding="utf-8") as f:
    f.write("ABIDE theorem-driven application summary\n")
    f.write("=======================================\n\n")
    f.write(f"Regions: {R}\n")
    f.write(f"Loop-free edges: {q}\n")
    f.write(f"Full symmetric pairs: {qfull}\n\n")
    for _,row in table1.iterrows():
        f.write(
            f"{row['geometry']}: numerical rank={int(row['numerical_rank'])}, "
            f"effective rank={row['effective_rank']:.2f}, "
            f"entropy rank={row['entropy_rank']:.2f}, "
            f"95% trace rank={int(row['rank_95pct_trace'])} "
            f"({100*row['fraction_dimension_95pct']:.2f}% of 6670).\n"
        )
    f.write("\n")
    f.write(
        f"Anatomical self-pair trace loss fraction: "
        f"{100*trace_loss_fraction:.3f}%.\n"
    )
    f.write(
        f"Generic normalized theorem bound 4/(R+1): "
        f"{100*bound:.3f}%.\n"
    )
    f.write(
        f"Maximum leading interlacing violation checked: "
        f"{max_violation:.3e}.\n"
    )

print("\nFinished.")
print("Results written to:")
print(OUT)
