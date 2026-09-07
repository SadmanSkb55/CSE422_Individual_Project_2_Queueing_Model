import os
import json
import csv
import math
import numpy as np
from scipy import stats

from random_generators import uniform, exponential, poisson
from mm1_queue import simulate_mm1

os.makedirs("data", exist_ok=True)

N        = 100_000
N_EVENTS = 300_000


# ── helpers ──────────────────────────────────────────────────────────────────

def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  JSON → {path}")

def write_csv(path, rows, headers):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  CSV  → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Q1 — Uniform
# ════════════════════════════════════════════════════════════════════════════
print("\n── Q1: Uniform ──")

samples_u = np.array(uniform(size=N))
x_vals    = np.round(np.linspace(0.75, 1.0, 100, endpoint=False), 6).tolist()
empirical = [round(float((samples_u > x).mean()), 8) for x in x_vals]
theory    = [round(1 - x, 8) for x in x_vals]
error     = [round(abs(e - t), 8) for e, t in zip(empirical, theory)]

q1 = {
    "description": "P(U > x) for U ~ Uniform(0,1), x in (0.75, 1)",
    "N": N,
    "columns": ["x", "empirical_P(U>x)", "theoretical_P(U>x)", "abs_error"],
    "data": [{"x": x, "empirical": e, "theoretical": t, "abs_error": err}
             for x, e, t, err in zip(x_vals, empirical, theory, error)]
}
write_json("data/q1_uniform.json", q1)
write_csv("data/q1_uniform.csv",
          list(zip(x_vals, empirical, theory, error)),
          ["x", "empirical_P(U>x)", "theoretical_P(U>x)", "abs_error"])


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Exponential
# ════════════════════════════════════════════════════════════════════════════
print("\n── Q2: Exponential (E[X]=10) ──")

EX        = 10.0
LAM_X     = 1.0 / EX
custom_exp = np.array(exponential(mean=EX, size=N))
x_exp      = np.round(np.linspace(0, 80, 200), 4).tolist()

# PDF (histogram density at each x)
counts, bin_edges = np.histogram(custom_exp, bins=200, density=True)
bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
pdf_empirical  = counts.tolist()
pdf_theoretical = [round(float(stats.expon.pdf(xv, scale=EX)), 8) for xv in bin_centers]

# Survival
surv_empirical   = [round(float((custom_exp > xv).mean()), 8) for xv in x_exp]
surv_theoretical = [round(float(math.exp(-LAM_X * xv)), 8) for xv in x_exp]
surv_error       = [round(abs(e - t), 8) for e, t in zip(surv_empirical, surv_theoretical)]

# Validation stats
ks_stat_exp, ks_p_exp = stats.kstest(custom_exp, 'expon', args=(0, EX))

q2_exp = {
    "description": "Exponential RV with E[X]=10 (lambda=0.1)",
    "N": N,
    "mean_expected": EX,
    "mean_custom":   round(float(custom_exp.mean()), 6),
    "std_expected":  EX,
    "std_custom":    round(float(custom_exp.std()), 6),
    "ks_statistic":  round(float(ks_stat_exp), 8),
    "ks_pvalue":     round(float(ks_p_exp), 8),
    "pdf": {
        "columns": ["x_bin_center", "pdf_empirical", "pdf_theoretical"],
        "data": [{"x": round(xv, 4), "pdf_empirical": round(pe, 8), "pdf_theoretical": round(pt, 8)}
                 for xv, pe, pt in zip(bin_centers, pdf_empirical, pdf_theoretical)]
    },
    "survival": {
        "columns": ["x", "empirical_P(X>x)", "theoretical_P(X>x)", "abs_error"],
        "data": [{"x": xv, "empirical": e, "theoretical": t, "abs_error": err}
                 for xv, e, t, err in zip(x_exp, surv_empirical, surv_theoretical, surv_error)]
    }
}
write_json("data/q2_exponential.json", q2_exp)

# CSV — pdf
write_csv("data/q2_exponential_pdf.csv",
          list(zip(bin_centers, pdf_empirical, pdf_theoretical)),
          ["x_bin_center", "pdf_empirical", "pdf_theoretical"])
# CSV — survival
write_csv("data/q2_exponential_survival.csv",
          list(zip(x_exp, surv_empirical, surv_theoretical, surv_error)),
          ["x", "empirical_P(X>x)", "theoretical_P(X>x)", "abs_error"])


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Poisson
# ════════════════════════════════════════════════════════════════════════════
print("\n── Q2: Poisson (E[Z]=10) ──")

EZ         = 10.0
custom_poi = np.array(poisson(mean=EZ, size=N))
z_vals     = list(range(0, 31))

pmf_empirical   = [round(float((custom_poi == z).mean()), 8) for z in z_vals]
pmf_theoretical = [round(float(stats.poisson.pmf(z, mu=EZ)), 8) for z in z_vals]
pmf_error       = [round(abs(e - t), 8) for e, t in zip(pmf_empirical, pmf_theoretical)]

surv_poi_empirical   = [round(float((custom_poi > z).mean()), 8) for z in z_vals]
surv_poi_theoretical = [round(float(1 - stats.poisson.cdf(z, mu=EZ)), 8) for z in z_vals]
surv_poi_error       = [round(abs(e - t), 8) for e, t in zip(surv_poi_empirical, surv_poi_theoretical)]

ks_stat_poi, ks_p_poi = stats.kstest(custom_poi, stats.poisson(EZ).cdf)

q2_poi = {
    "description": "Poisson RV with E[Z]=10 (lambda=10)",
    "N": N,
    "mean_expected": EZ,
    "mean_custom":   round(float(custom_poi.mean()), 6),
    "var_expected":  EZ,
    "var_custom":    round(float(custom_poi.var()), 6),
    "ks_statistic":  round(float(ks_stat_poi), 8),
    "ks_pvalue":     round(float(ks_p_poi), 8),
    "pmf": {
        "columns": ["z", "pmf_empirical", "pmf_theoretical", "abs_error"],
        "data": [{"z": z, "pmf_empirical": e, "pmf_theoretical": t, "abs_error": err}
                 for z, e, t, err in zip(z_vals, pmf_empirical, pmf_theoretical, pmf_error)]
    },
    "survival": {
        "columns": ["z", "empirical_P(Z>z)", "theoretical_P(Z>z)", "abs_error"],
        "data": [{"z": z, "empirical": e, "theoretical": t, "abs_error": err}
                 for z, e, t, err in zip(z_vals, surv_poi_empirical, surv_poi_theoretical, surv_poi_error)]
    }
}
write_json("data/q2_poisson.json", q2_poi)

write_csv("data/q2_poisson_pmf.csv",
          list(zip(z_vals, pmf_empirical, pmf_theoretical, pmf_error)),
          ["z", "pmf_empirical", "pmf_theoretical", "abs_error"])
write_csv("data/q2_poisson_survival.csv",
          list(zip(z_vals, surv_poi_empirical, surv_poi_theoretical, surv_poi_error)),
          ["z", "empirical_P(Z>z)", "theoretical_P(Z>z)", "abs_error"])


# ════════════════════════════════════════════════════════════════════════════
# Q3 — M/M/1 Pn per rho
# ════════════════════════════════════════════════════════════════════════════

def analytical_pn(rho, n):
    return round((1 - rho) * (rho ** n), 8)

configs = [
    (0.90, 9.0,  10.0, "rho09",  "q3_pn_rho09"),
    (0.50, 5.0,  10.0, "rho05",  "q3_pn_rho05"),
    (0.25, 2.5,  10.0, "rho025", "q3_pn_rho025"),
]

n_range   = list(range(0, 31))
q3_summary = {"description": "M/M/1 stationary probabilities Pn", "rhos": []}

for rho, lam, mu, key, fname in configs:
    print(f"\n── Q3: M/M/1 ρ={rho} (λ={lam}, μ={mu}) ──")

    pn_sim = simulate_mm1(lam=lam, mu=mu, n_events=N_EVENTS)

    analytical = [analytical_pn(rho, n) for n in n_range]
    simulated  = [round(float(pn_sim.get(n, 0.0)), 8) for n in n_range]
    sim_error  = [round(abs(s - a), 8) for s, a in zip(simulated, analytical)]
    mae        = round(float(np.mean(sim_error)), 8)

    block = {
        "rho": rho, "lambda": lam, "mu": mu,
        "n_events": N_EVENTS,
        "MAE_simulation_vs_analytical": mae,
        "columns": ["n", "analytical_Pn", "simulated_Pn", "abs_error"],
        "data": [{"n": n, "analytical": a, "simulated": s, "abs_error": e}
                 for n, a, s, e in zip(n_range, analytical, simulated, sim_error)]
    }

    write_json(f"data/{fname}.json", block)
    write_csv(f"data/{fname}.csv",
              list(zip(n_range, analytical, simulated, sim_error)),
              ["n", "analytical_Pn", "simulated_Pn", "abs_error"])

    q3_summary["rhos"].append(block)

write_json("data/q3_summary.json", q3_summary)
print("  JSON → data/q3_summary.json  (all rhos combined)")


# ════════════════════════════════════════════════════════════════════════════
# Validation stats summary
# ════════════════════════════════════════════════════════════════════════════

val_stats = {
    "exponential": {
        "E[X]": EX,
        "N": N,
        "custom_mean": round(float(custom_exp.mean()), 6),
        "custom_std":  round(float(custom_exp.std()), 6),
        "mean_error":  round(abs(custom_exp.mean() - EX), 6),
        "std_error":   round(abs(custom_exp.std()  - EX), 6),
        "ks_statistic": round(float(ks_stat_exp), 8),
        "ks_pvalue":    round(float(ks_p_exp), 8),
        "ks_pass":      bool(ks_p_exp > 0.05)
    },
    "poisson": {
        "E[Z]": EZ,
        "N": N,
        "custom_mean": round(float(custom_poi.mean()), 6),
        "custom_var":  round(float(custom_poi.var()), 6),
        "mean_error":  round(abs(custom_poi.mean() - EZ), 6),
        "var_error":   round(abs(custom_poi.var()  - EZ), 6),
        "ks_statistic": round(float(ks_stat_poi), 8),
        "ks_pvalue":    round(float(ks_p_poi), 8),
        "ks_pass":      bool(ks_p_poi > 0.05)
    }
}
write_json("data/validation_stats.json", val_stats)

print("\n✓ All data exported to data/")
