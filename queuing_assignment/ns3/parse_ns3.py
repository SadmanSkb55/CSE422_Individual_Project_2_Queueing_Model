import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mm1_queue import simulate_mm1


def parse_ns3_output(filepath):
    pn = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('n') or not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                pn[int(parts[0])] = float(parts[1])
    return pn


def analytical_pn(rho, n):
    return (1 - rho) * (rho ** n)


def rho_to_slug(rho):
    return str(rho).replace("0.", "").replace(".", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--rho',   type=float, default=0.9)
    parser.add_argument('--lam',   type=float, default=9.0)
    parser.add_argument('--mu',    type=float, default=10.0)
    args = parser.parse_args()

    n_range = np.arange(0, 31)

    # ── NS-3 ─────────────────────────────────────────────────────────────────
    pn_ns3     = parse_ns3_output(args.input)
    pn_ns3_arr = np.array([pn_ns3.get(n, 0.0) for n in n_range])

    # ── Python simulation ─────────────────────────────────────────────────────
    print(f"Running Python M/M/1 simulation (λ={args.lam}, μ={args.mu}) ...")
    pn_sim     = simulate_mm1(lam=args.lam, mu=args.mu, n_events=300_000)
    pn_sim_arr = np.array([pn_sim.get(n, 0.0) for n in n_range])

    # ── Analytical ────────────────────────────────────────────────────────────
    pn_anal_arr = analytical_pn(args.rho, n_range)

    # ── Plot ──────────────────────────────────────────────────────────────────
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(n_range - width, pn_anal_arr, width=width, alpha=0.75,
           color='steelblue', label='Analytical: Pₙ = (1−ρ)ρⁿ')
    ax.bar(n_range,          pn_sim_arr, width=width, alpha=0.75,
           color='tomato',    label='Python Simulation')
    ax.bar(n_range + width,  pn_ns3_arr, width=width, alpha=0.75,
           color='seagreen',  label='NS-3 Simulation')

    ax.set_xlabel('n  (number of customers in system)')
    ax.set_ylabel('Pₙ')
    ax.set_title(f'Q3 — M/M/1 Pₙ: Analytical vs Python vs NS-3  '
                 f'(λ={args.lam}, μ={args.mu}, ρ={args.rho})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-1, 31)

    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'plots'), exist_ok=True)

    slug    = rho_to_slug(args.rho)
    outname = f'q3_mm1_with_ns3_rho{slug}.png'
    out     = os.path.join(os.path.dirname(__file__), '..', 'plots', outname)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved → {os.path.abspath(out)}")
    plt.close()


if __name__ == '__main__':
    main()