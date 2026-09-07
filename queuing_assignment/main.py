import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from random_generators import uniform, exponential, poisson
from mm1_queue import simulate_mm1
from validation import validate_exponential, validate_poisson

os.makedirs("plots", exist_ok=True)

N = 100_000  # sample size for Q1 and Q2


# ════════════════════════════════════════════════════════════════════════════
# Q1 — Uniform Random Variable
# ════════════════════════════════════════════════════════════════════════════

print("\n── Q1: Uniform ──────────────────────────────────────────")

samples_u   = np.array(uniform(size=N))
x_vals      = np.linspace(0.75, 1.0, 500, endpoint=False)
empirical_u = np.array([(samples_u > x).mean() for x in x_vals])
theoretical_u = 1 - x_vals

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, theoretical_u, color='steelblue', linewidth=2.5,
        label='Theoretical: P(U > x) = 1 − x')
ax.plot(x_vals, empirical_u,   color='tomato',    linewidth=1.8,
        linestyle='--', label=f'Empirical (N={N:,})')
ax.set_xlabel('x')
ax.set_ylabel('P(U > x)')
ax.set_title('Q1 — Uniform(0,1): Survival Function for x ∈ (0.75, 1)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/q1_uniform.png', dpi=150)
plt.close()
print("  Saved → plots/q1_uniform.png")


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Exponential
# ════════════════════════════════════════════════════════════════════════════

print("\n── Q2: Exponential (E[X]=10) ────────────────────────────")

EX   = 10.0
LAM_X = 1.0 / EX
custom_exp  = np.array(exponential(mean=EX, size=N))
scipy_exp   = stats.expon.rvs(scale=EX, size=N)
x_exp       = np.linspace(0, 80, 500)

# PDF comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(custom_exp, bins=80, density=True, alpha=0.5, color='tomato',    label='Custom Exponential')
ax.hist(scipy_exp,  bins=80, density=True, alpha=0.5, color='steelblue', label='Scipy Exponential')
ax.plot(x_exp, stats.expon.pdf(x_exp, scale=EX), 'k-', linewidth=2, label='Theoretical PDF')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Q2 — Exponential PDF (E[X] = 10)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/q2_exponential_pdf.png', dpi=150)
plt.close()
print("  Saved → plots/q2_exponential_pdf.png")

# Survival P(X > x)
empirical_exp_surv   = np.array([(custom_exp > x).mean() for x in x_exp])
theoretical_exp_surv = np.exp(-LAM_X * x_exp)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_exp, theoretical_exp_surv, color='steelblue', linewidth=2.5, label='Theoretical: e^(−λx)')
ax.plot(x_exp, empirical_exp_surv,   color='tomato',    linewidth=1.8,
        linestyle='--', label='Empirical (Custom)')
ax.set_xlabel('x')
ax.set_ylabel('P(X > x)')
ax.set_title('Q2 — Exponential Survival Function (E[X] = 10)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/q2_exponential_survival.png', dpi=150)
plt.close()
print("  Saved → plots/q2_exponential_survival.png")


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Poisson
# ════════════════════════════════════════════════════════════════════════════

print("\n── Q2: Poisson (E[Z]=10) ────────────────────────────────")

EZ = 10.0
custom_poi = np.array(poisson(mean=EZ, size=N))
scipy_poi  = stats.poisson.rvs(mu=EZ, size=N)
z_vals     = np.arange(0, 31)

# PMF comparison
fig, ax = plt.subplots(figsize=(8, 5))
custom_pmf = np.array([(custom_poi == z).mean() for z in z_vals])
scipy_pmf  = np.array([(scipy_poi  == z).mean() for z in z_vals])
theory_pmf = stats.poisson.pmf(z_vals, mu=EZ)

ax.bar(z_vals - 0.3, custom_pmf, width=0.28, alpha=0.7, color='tomato',    label='Custom Poisson')
ax.bar(z_vals,       scipy_pmf,  width=0.28, alpha=0.7, color='steelblue', label='Scipy Poisson')
ax.plot(z_vals, theory_pmf, 'ko--', markersize=4, linewidth=1.5,            label='Theoretical PMF')
ax.set_xlabel('z')
ax.set_ylabel('P(Z = z)')
ax.set_title('Q2 — Poisson PMF (E[Z] = 10)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/q2_poisson_pmf.png', dpi=150)
plt.close()
print("  Saved → plots/q2_poisson_pmf.png")

# Survival P(Z > z)
empirical_poi_surv   = np.array([(custom_poi > z).mean() for z in z_vals])
theoretical_poi_surv = 1 - stats.poisson.cdf(z_vals, mu=EZ)

fig, ax = plt.subplots(figsize=(8, 5))
ax.step(z_vals, theoretical_poi_surv, color='steelblue', linewidth=2.5, label='Theoretical: 1 − CDF(z)')
ax.step(z_vals, empirical_poi_surv,   color='tomato',    linewidth=1.8,
        linestyle='--', label='Empirical (Custom)')
ax.set_xlabel('z')
ax.set_ylabel('P(Z > z)')
ax.set_title('Q2 — Poisson Survival Function (E[Z] = 10)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/q2_poisson_survival.png', dpi=150)
plt.close()
print("  Saved → plots/q2_poisson_survival.png")


# ════════════════════════════════════════════════════════════════════════════
# Q2 — Validation summary (KS tests)
# ════════════════════════════════════════════════════════════════════════════

print()
validate_exponential(mean=EX, n=N)
validate_poisson(mean=EZ, n=N)


# ════════════════════════════════════════════════════════════════════════════
# Q3 — M/M/1 Queue: lambda=9, mu=10 (rho=0.9)
# ════════════════════════════════════════════════════════════════════════════

print("\n── Q3: M/M/1 simulation (λ=9, μ=10, ρ=0.9) ─────────────")

LAM  = 9
MU   = 10
RHO  = LAM / MU
N_EVENTS = 300_000

pn_sim  = simulate_mm1(lam=LAM, mu=MU, n_events=N_EVENTS)
n_range = np.arange(0, 31)

def analytical_pn(rho, n):
    return (1 - rho) * (rho ** n)

pn_analytical = analytical_pn(RHO, n_range)
pn_simulated  = np.array([pn_sim.get(n, 0.0) for n in n_range])

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(n_range - 0.2, pn_analytical, width=0.38, alpha=0.75,
       color='steelblue', label='Analytical: Pₙ = (1−ρ)ρⁿ')
ax.bar(n_range + 0.2, pn_simulated,  width=0.38, alpha=0.75,
       color='tomato',    label=f'Simulated ({N_EVENTS:,} events)')
ax.set_xlabel('n  (number of customers in system)')
ax.set_ylabel('Pₙ')
ax.set_title(f'Q3 — M/M/1 Stationary Probabilities  (λ={LAM}, μ={MU}, ρ={RHO})')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-1, 31)
plt.tight_layout()
plt.savefig('plots/q3_mm1.png', dpi=150)
plt.close()
print("  Saved → plots/q3_mm1.png")


# ════════════════════════════════════════════════════════════════════════════
# Q3 — Compare rho = 0.9, 0.5, 0.25
# ════════════════════════════════════════════════════════════════════════════

print("\n── Q3: Pₙ comparison across ρ = 0.9, 0.5, 0.25 ─────────")

configs = [
    (0.90, 'tomato',    9.0,  10.0),
    (0.50, 'steelblue', 5.0,  10.0),
    (0.25, 'seagreen',  2.5,  10.0),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for rho, color, lam, mu in configs:
    print(f"  Simulating ρ={rho} (λ={lam}, μ={mu}) ...")
    pn_s = simulate_mm1(lam=lam, mu=mu, n_events=N_EVENTS)
    pn_sim_arr  = np.array([pn_s.get(n, 0.0) for n in n_range])
    pn_anal_arr = analytical_pn(rho, n_range)

    axes[0].plot(n_range, pn_anal_arr, color=color, linewidth=2,
                 marker='o', markersize=3, label=f'ρ={rho}')
    axes[1].plot(n_range, pn_sim_arr,  color=color, linewidth=2,
                 marker='s', markersize=3, linestyle='--', label=f'ρ={rho}')

axes[0].set_title('Analytical Pₙ')
axes[1].set_title('Simulated Pₙ')
for ax in axes:
    ax.set_xlabel('n')
    ax.set_ylabel('Pₙ')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-1, 31)

fig.suptitle('Q3 — M/M/1: Pₙ for ρ = 0.9, 0.5, 0.25', fontsize=14)
plt.tight_layout()
plt.savefig('plots/q3_rho_comparison.png', dpi=150)
plt.close()
print("  Saved → plots/q3_rho_comparison.png")

print("\n✓ All done. Check the plots/ folder.")
