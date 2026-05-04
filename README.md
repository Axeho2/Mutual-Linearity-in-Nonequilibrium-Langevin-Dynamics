# Mutual Linearity Simulation Codes

This folder contains Monte Carlo simulation and plotting scripts for the F1-ATPase tilted-periodic-potential model used to test steady-state and non-stationary mutual linearity in overdamped Langevin dynamics.

The code is split into simulation scripts, plotting scripts, and one convenience shell script that runs the full workflow.

## Files

### `simulation_NESS.py`
Runs the steady-state simulations and saves the sampled observables to `.npz` files.

It scans the local Gaussian perturbation strength `lambda` for a chosen perturbation width and records:

- sector occupations: `occ_A`, `occ_B`, `occ_C`
- smoothed density probes: `rho_A`, `rho_B`
- local input power: `power_B`
- angular velocity: `omega`
- total input power: `total_power`
- local entropy production in the sector-B window: `sigma_B`

Output files are saved to:

```bash
data/<width>_mutual_linearity.npz
```

For example:

```bash
data/0.05_mutual_linearity.npz
```

### `plot_NESS.py`
Reads the steady-state `.npz` files and generates the mutual-linearity figures.

It creates one observable-vs-lambda plot and four mutual-linearity plots for each width. For `width = 1.0`, the plotting code connects the data points to highlight nonlinear behavior and suppresses the `R^2` and slope annotations.

Figures are saved to:

```bash
figures/
```

Each figure is saved as both `.pdf` and `.svg`.

### `simulation_nonstationary.py`
Runs non-stationary Laplace-domain simulations and saves the sampled Laplace-transformed observables to `.npz` files.

The default non-stationary simulation uses:

```python
omegas = [2.0, 5.0, 10.0]
T_cut = 5 s
```

The Laplace integral is directly truncated at `T_cut`; no stationary-tail correction is added.

Output files are saved to:

```bash
data/nonstationary_<width>_laplace_mutual_linearity.npz
```

For example:

```bash
data/nonstationary_0.1_laplace_mutual_linearity.npz
```

### `plot_nonstationary.py`
Reads the non-stationary `.npz` files and generates Laplace-domain mutual-linearity figures.

Each mutual-linearity panel shows all default frequencies on the same axes. Figures are saved as both `.pdf` and `.svg` in `figures/`.

### `run_all_simulations.sh`
Convenience script for macOS/Linux. It runs the full workflow:

1. steady-state simulations for widths `0.05`, `0.5`, and `1.0`
2. steady-state plotting
3. non-stationary simulation for width `0.1`
4. non-stationary plotting

## Python dependencies

The scripts require:

- Python 3.9 or newer
- `numpy`
- `matplotlib`
- `numba`

Install them with:

```bash
python3 -m pip install numpy matplotlib numba
```

Optional but recommended on macOS: create a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib numba
```

## Running the full workflow

From the folder containing the scripts, run:

```bash
chmod +x run_all_simulations.sh
./run_all_simulations.sh
```

By default, the simulation scripts reuse existing data files if they already exist. To force recomputation and overwrite cached results, run:

```bash
./run_all_simulations.sh --force
```

You can choose a different Python executable by setting `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python3 ./run_all_simulations.sh
```

## Running scripts individually

### Steady-state simulation

```bash
python3 simulation_NESS.py --broad_band 0.05
python3 simulation_NESS.py --broad_band 0.5
python3 simulation_NESS.py --broad_band 1.0
```

Useful options:

```bash
python3 simulation_NESS.py --broad_band 0.05 --force
python3 simulation_NESS.py --broad_band 0.05 --lambda_min -50 --lambda_max 0 --n_lambdas 10
python3 simulation_NESS.py --broad_band 0.05 --quick
```

### Steady-state plotting

```bash
python3 plot_NESS.py --widths 0.05 0.5 1.0
```

If some data files are missing and you want to skip them rather than stop:

```bash
python3 plot_NESS.py --widths 0.05 0.5 1.0 --allow_missing
```

### Non-stationary simulation

```bash
python3 simulation_nonstationary.py --band_width 0.1
```

Useful options:

```bash
python3 simulation_nonstationary.py --band_width 0.1 --force
python3 simulation_nonstationary.py --debug
```

### Non-stationary plotting

```bash
python3 plot_nonstationary.py --band_width 0.1
```

If the data file is missing and you want the script to exit without raising an error:

```bash
python3 plot_nonstationary.py --band_width 0.1 --allow_missing
```

## Output structure

After running the scripts, the directory structure will look like:

```bash
.
├── data/
│   ├── 0.05_mutual_linearity.npz
│   ├── 0.5_mutual_linearity.npz
│   ├── 1.0_mutual_linearity.npz
│   └── nonstationary_0.1_laplace_mutual_linearity.npz
├── figures/
│   ├── *.pdf
│   └── *.svg
├── simulation_NESS.py
├── plot_NESS.py
├── simulation_nonstationary.py
├── plot_nonstationary.py
└── run_all_simulations.sh
```

## Notes

- The production simulations are computationally expensive because they use many trajectories and long sampling windows.
- The first run may spend extra time compiling `numba` functions. Later runs are usually faster because compilation results are cached.
- Existing data files are not overwritten unless `--force` is supplied.
- The plotting scripts only read saved data; they do not run simulations.
- The `.npz` storage format is intentionally kept stable so that figures can be regenerated without rerunning simulations.
