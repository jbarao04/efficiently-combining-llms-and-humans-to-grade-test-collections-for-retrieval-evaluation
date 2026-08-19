# Orthogonality test: does synonym stability add unique variance beyond permutation stability?

import os
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import csv
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


#Load data (pure csv, no pandas)
def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


b1b_rows = load_csv("results/level2/b1b_features.csv")
b1a_rows = load_csv("results/level2/b1a_features.csv")

# Try v4, v3, v2
pq_rows = None
for v in [4, 3, 2]:
    path = f"results/level2/per_query_features_v{v}.csv"
    if os.path.exists(path):
        pq_rows = load_csv(path)
        print(f"Loaded per_query_features_v{v}.csv")
        break
assert pq_rows is not None, "No per_query_features file found"

# Build lookup dicts
b1b_dict = {r["query_id"]: float(r["b1b_stability_tau"]) for r in b1b_rows}
b1a_dict = {r["query_id"]: float(r["b1a_stability_tau"]) for r in b1a_rows}

# Find spearman and fisher columns
pq_keys = list(pq_rows[0].keys())
spearman_col = None
for c in ["spearman", "spearman_rho", "rho", "per_query_spearman"]:
    if c in pq_keys:
        spearman_col = c
        break
assert spearman_col, f"No spearman column found. Columns: {pq_keys}"

fisher_col = None
for c in ["fisher_ratio", "fisher", "fisher_score"]:
    if c in pq_keys:
        fisher_col = c
        break
assert fisher_col, f"No fisher column found. Columns: {pq_keys}"

pq_dict = {r["query_id"]: (float(r[fisher_col]), float(r[spearman_col])) for r in pq_rows}

# Merge
common_ids = sorted(set(b1b_dict) & set(b1a_dict) & set(pq_dict))
b1a_tau = np.array([b1a_dict[qid] for qid in common_ids])
b1b_tau = np.array([b1b_dict[qid] for qid in common_ids])
fisher = np.array([pq_dict[qid][0] for qid in common_ids])
spearman = np.array([pq_dict[qid][1] for qid in common_ids])

print(f"Merged queries: {len(common_ids)} (expected 308)")
assert len(common_ids) == 308, f"Expected 308 queries, got {len(common_ids)}"


# ── Helper functions ───────────────────────────────────────────────────────────
def ols_r2(X, y):
    """R² from OLS regression using numpy. X shape (n,) or (n, p)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Add intercept
    X_i = np.column_stack([np.ones(len(X)), X])
    # Solve normal equations
    beta, _, _, _ = np.linalg.lstsq(X_i, y, rcond=None)
    y_hat = X_i @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def ols_predict(X, y):
    """Return fitted values from OLS."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_i = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_i, y, rcond=None)
    return X_i @ beta


def partial_corr(x, y, covariates):
    """Pearson partial correlation: r(x, y | covariates)."""
    covariates = np.asarray(covariates)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)
    rx = x - ols_predict(covariates, x)
    ry = y - ols_predict(covariates, y)
    r, p = stats.pearsonr(rx, ry)
    return r, p


# ── 1. Pairwise correlations ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. PAIRWISE PEARSON CORRELATION MATRIX")
print("=" * 60)

labels = ["b1a_tau", "b1b_tau", "fisher", "spearman"]
data = np.column_stack([b1a_tau, b1b_tau, fisher, spearman])
corr = np.corrcoef(data, rowvar=False)

print(f"{'':>12s}", end="")
for l in labels:
    print(f"{l:>12s}", end="")
print()
for i, l in enumerate(labels):
    print(f"{l:>12s}", end="")
    for j in range(len(labels)):
        print(f"{corr[i,j]:>12.4f}", end="")
    print()

r_b1a_b1b = corr[0, 1]
print(f"\nr(B1a, B1b) = {r_b1a_b1b:.4f}")
if r_b1a_b1b > 0.7:
    print("  -> Above 0.7: they measure largely the same construct")
elif r_b1a_b1b > 0.5:
    print("  -> Between 0.5-0.7: moderate overlap, some independence")
else:
    print("  -> Below 0.5: genuinely different axes")
print(f"  (cf. r(B3, B1b) = 0.724 from thesis)")


# ── 2. Kendall tau of B1a with Spearman ──────────────────────────────────────
print("\n" + "=" * 60)
print("2. KENDALL TAU: B1a vs Spearman (with bootstrap CI)")
print("=" * 60)

tau_obs, p_tau = stats.kendalltau(b1a_tau, spearman)
print(f"Kendall tau(B1a, Spearman) = {tau_obs:.4f}, p = {p_tau:.2e}")

# Bootstrap 95% CI
rng = np.random.RandomState(42)
n = len(b1a_tau)
boot_taus = np.empty(10000)
for i in range(10000):
    idx = rng.randint(0, n, size=n)
    boot_taus[i] = stats.kendalltau(b1a_tau[idx], spearman[idx]).statistic

ci_lo, ci_hi = np.percentile(boot_taus, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  (expected ~0.169)")


# ── 3. Partial correlations ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PARTIAL CORRELATIONS")
print("=" * 60)

partials = [
    ("r(B1a, spearman | B1b)",          b1a_tau, spearman, b1b_tau.reshape(-1, 1)),
    ("r(B1a, spearman | Fisher)",       b1a_tau, spearman, fisher.reshape(-1, 1)),
    ("r(B1a, spearman | Fisher, B1b)",  b1a_tau, spearman, np.column_stack([fisher, b1b_tau])),
    ("r(B1b, spearman | B1a)",          b1b_tau, spearman, b1a_tau.reshape(-1, 1)),
    ("r(B1b, spearman | Fisher, B1a)",  b1b_tau, spearman, np.column_stack([fisher, b1a_tau])),
]

for label, x, y, cov in partials:
    r, p = partial_corr(x, y, cov)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {label:40s} = {r:+.4f}  (p = {p:.2e}) {sig}")

print("\n  Critical: r(B1a, spearman | Fisher, B1b) -- if near zero, B1a is subsumed.")


# ── 4. Commonality analysis ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. COMMONALITY ANALYSIS: Fisher + B1a + B1b -> Spearman")
print("=" * 60)

# a=Fisher, b=B1a, c=B1b
r2_a   = ols_r2(fisher, spearman)
r2_b   = ols_r2(b1a_tau, spearman)
r2_c   = ols_r2(b1b_tau, spearman)
r2_ab  = ols_r2(np.column_stack([fisher, b1a_tau]), spearman)
r2_ac  = ols_r2(np.column_stack([fisher, b1b_tau]), spearman)
r2_bc  = ols_r2(np.column_stack([b1a_tau, b1b_tau]), spearman)
r2_abc = ols_r2(np.column_stack([fisher, b1a_tau, b1b_tau]), spearman)

print(f"\nSubset R2 values:")
print(f"  R2(Fisher)           = {r2_a:.6f}")
print(f"  R2(B1a)              = {r2_b:.6f}")
print(f"  R2(B1b)              = {r2_c:.6f}")
print(f"  R2(Fisher, B1a)      = {r2_ab:.6f}")
print(f"  R2(Fisher, B1b)      = {r2_ac:.6f}")
print(f"  R2(B1a, B1b)         = {r2_bc:.6f}")
print(f"  R2(Fisher, B1a, B1b) = {r2_abc:.6f}")

# Unique contributions
u_a = r2_abc - r2_bc
u_b = r2_abc - r2_ac
u_c = r2_abc - r2_ab

# Common all three
c_abc = r2_a + r2_b + r2_c - r2_ab - r2_ac - r2_bc + r2_abc

# Common pairs only (excluding the triple)
c_ab_only = (r2_a + r2_b - r2_ab) - c_abc
c_ac_only = (r2_a + r2_c - r2_ac) - c_abc
c_bc_only = (r2_b + r2_c - r2_bc) - c_abc

components = {
    "Unique Fisher":            u_a,
    "Unique B1a":               u_b,
    "Unique B1b":               u_c,
    "Common Fisher & B1a only": c_ab_only,
    "Common Fisher & B1b only": c_ac_only,
    "Common B1a & B1b only":    c_bc_only,
    "Common all three":         c_abc,
}

total = sum(components.values())
print(f"\nCommonality decomposition (total R2 = {r2_abc:.6f}, sum = {total:.6f}):")
print(f"  {'Component':<30s} {'R2':>10s} {'% of total':>10s}")
print(f"  {'-'*50}")
for name, val in components.items():
    pct = 100 * val / r2_abc if r2_abc > 0 else 0
    print(f"  {name:<30s} {val:>10.4f} {pct:>9.1f}%")


# ── 5. Compare to B3 decomposition ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SIDE-BY-SIDE: Fisher+B3+B1b vs Fisher+B1a+B1b")
print("=" * 60)

print(f"\n  {'Component':<30s} {'Fisher+B3+B1b':>18s} {'Fisher+B1a+B1b':>18s}")
print(f"  {'-'*68}")
comparisons = [
    ("Unique Fisher",    0.0377, u_a),
    ("Unique B3 / B1a",  0.0003, u_b),
    ("Unique B1b",       0.0604, u_c),
    ("Common all",       0.1044, c_abc),
    ("Total R2",         0.2592, r2_abc),
]
for name, b3_val, b1a_val in comparisons:
    b3_pct = f"{b3_val:.4f} ({100*b3_val/0.2592:.1f}%)"
    b1a_pct = f"{b1a_val:.4f} ({100*b1a_val/r2_abc:.1f}%)"
    print(f"  {name:<30s} {b3_pct:>18s} {b1a_pct:>18s}")


# ── 6. Incremental R2 tests ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. INCREMENTAL R2 TEST: Adding B1a to Fisher + B1b")
print("=" * 60)

r2_base = r2_ac  # Fisher + B1b
r2_full = r2_abc  # Fisher + B1a + B1b
delta_r2 = r2_full - r2_base
n_obs = len(common_ids)
p_full = 3
p_base = 2
df_num = p_full - p_base  # = 1
df_den = n_obs - p_full - 1

f_stat = (delta_r2 / df_num) / ((1 - r2_full) / df_den)
p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)

print(f"  R2(Fisher + B1b)       = {r2_base:.6f}")
print(f"  R2(Fisher + B1b + B1a) = {r2_full:.6f}")
print(f"  Delta R2               = {delta_r2:.6f}")
print(f"  F({df_num}, {df_den})               = {f_stat:.4f}")
print(f"  p-value                = {p_value:.2e}")


# ── 7. Summary / Verdict ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. VERDICT")
print("=" * 60)

b1a_unique_pct = 100 * u_b / r2_abc if r2_abc > 0 else 0

print(f"""
  r(B1a, B1b)              = {r_b1a_b1b:.4f}
  B1a unique variance      = {u_b:.4f} ({b1a_unique_pct:.1f}% of total R2)
  B3  unique variance      = 0.0003 (0.1% of total R2) [known]
  Increment beyond F+B1b   : delta R2 = {delta_r2:.6f}, p = {p_value:.2e}
""")

if u_b > 0.01 * r2_abc and p_value < 0.05:
    print("  CONCLUSION: B1a carries independent information.")
    print("  The word-level axis is worth pursuing with a stronger operator.")
elif p_value < 0.05:
    print("  CONCLUSION: B1a is statistically significant but its unique")
    print(f"  contribution ({b1a_unique_pct:.1f}%) is very small -- marginal value.")
else:
    print("  CONCLUSION: B1a is subsumed like B3.")
    print("  Synonym-based perturbation is the wrong direction.")
