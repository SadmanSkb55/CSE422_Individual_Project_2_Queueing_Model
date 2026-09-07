# CSE422 / CSE562 — Individual Project 2: Queueing Theory Modelling and Simulation

**North South University** | Department of ECE | Course: Modelling and Simulation

This project builds a complete M/M/1 queueing simulation from scratch — starting from a custom Uniform random variable generator, deriving Exponential and Poisson generators from it, simulating the M/M/1 queue using discrete-event simulation, and finally validating all results against the exact analytical formula and NS-3.

---

## What This Project Does

| Part | What we do |
|---|---|
| **Q1** | Generate `U ~ Uniform(0,1)` from scratch and plot `P(U > x)` for `x ∈ (0.75, 1)` against the theoretical `1 - x` |
| **Q2** | Build `Exponential` (inverse CDF method) and `Poisson` (Knuth algorithm) generators from `U`; validate against `scipy.stats` using KS tests and plot PDFs and survival functions |
| **Q3** | Simulate M/M/1 queue using discrete-event simulation; compare stationary probabilities `Pₙ = (1−ρ)ρⁿ` across analytical, Python simulation, and NS-3 simulation for ρ = 0.9, 0.5, 0.25 |

---

## Project Structure

```
queuing_assignment/
├── main.py                  # Runs Q1 + Q2 + Q3, saves all plots
├── random_generators.py     # uniform(), exponential(), poisson() from scratch
├── mm1_queue.py             # Discrete-event M/M/1 simulator
├── validation.py            # KS tests and summary stats for Q2
├── export_data.py           # Exports all results to JSON + CSV in data/
├── plots/                   # All output figures (auto-created)
├── data/                    # All output data files (auto-created)
└── ns3/
    ├── mm1.cc               # NS-3 simulation (C++)
    └── parse_ns3.py         # Parses NS-3 output and plots 3-way comparison
```

---

## Requirements

- Python 3.9+
- NS-3.41 (for the NS-3 part only)

---

## How to Reproduce

### 1. Clone the repo

```bash
git clone https://github.com/SadmanSkb55/CSE422_Individual_Project_2_Queueing_Model.git
cd CSE422_Individual_Project_2_Queueing_Model
```

### 2. Set up a virtual environment (Windows)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Run the Python simulation (Q1 + Q2 + Q3)

```powershell
python main.py
```

Saves 6 plots to `plots/`.

### 4. Export all data to JSON and CSV

```powershell
python export_data.py
```

Saves all simulation and analytical values to `data/`.

---

### 5. NS-3 Part (requires WSL2 on Windows)

#### Install WSL2 and NS-3 (one-time setup)

```powershell
# In PowerShell (Admin)
wsl --install
```

Then inside WSL (Ubuntu):

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y g++ python3 cmake ninja-build git libboost-all-dev libgsl-dev libsqlite3-dev

cd ~
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3
cd ns-3
git checkout ns-3.41
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

#### Run NS-3 simulations

```bash
# Copy the simulation file
cp /mnt/d/queuing_assignment/ns3/mm1.cc ~/ns-3/scratch/mm1.cc
cd ~/ns-3

# rho = 0.9
./ns3 run scratch/mm1 > /mnt/d/queuing_assignment/ns3/ns3_output.txt

# rho = 0.5
./ns3 run "scratch/mm1 --lambda=5 --mu=10" > /mnt/d/queuing_assignment/ns3/ns3_output_rho05.txt

# rho = 0.25
./ns3 run "scratch/mm1 --lambda=2.5 --mu=10" > /mnt/d/queuing_assignment/ns3/ns3_output_rho025.txt
```

#### Parse NS-3 output and generate comparison plots

Back in Windows (venv active):

```powershell
python ns3/parse_ns3.py --input ns3/ns3_output.txt         --rho 0.9  --lam 9.0
python ns3/parse_ns3.py --input ns3/ns3_output_rho05.txt   --rho 0.5  --lam 5.0
python ns3/parse_ns3.py --input ns3/ns3_output_rho025.txt  --rho 0.25 --lam 2.5
```

Produces three plots in `plots/`:
- `q3_mm1_with_ns3_rho9.png`
- `q3_mm1_with_ns3_rho5.png`
- `q3_mm1_with_ns3_rho25.png`

Each shows the three-way comparison: **Analytical vs Python simulation vs NS-3**.

---

## Output Plots

| File | Content |
|---|---|
| `q1_uniform.png` | P(U > x) empirical vs theoretical |
| `q2_exponential_pdf.png` | Exponential PDF comparison |
| `q2_exponential_survival.png` | Exponential survival function |
| `q2_poisson_pmf.png` | Poisson PMF comparison |
| `q2_poisson_survival.png` | Poisson survival function |
| `q3_mm1.png` | M/M/1 Pₙ: analytical vs Python (ρ=0.9) |
| `q3_rho_comparison.png` | Pₙ overlay for ρ = 0.9, 0.5, 0.25 |
| `q3_mm1_with_ns3_rho9.png` | 3-way comparison, ρ=0.9 |
| `q3_mm1_with_ns3_rho5.png` | 3-way comparison, ρ=0.5 |
| `q3_mm1_with_ns3_rho25.png` | 3-way comparison, ρ=0.25 |