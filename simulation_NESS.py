"""
Steady-state mutual linearity simulation for the F1-ATPase tilted-periodic-potential model.

This file only runs the Monte Carlo simulation and saves data.  The sampling
method and NPZ storage format are kept the same as in the original combined
script.  Plotting is handled separately by plot.py.

Model on the unwrapped angular coordinate theta:

    d theta = mu * [3 U0 sin(3 theta) + F_drive
                    + lam * g_l(theta - z0)] dt
              + sqrt(2 mu T) dW_t .

Here g_l is an area-normalized periodic Gaussian packet with width local_width.
The integrator is the Euler-Maruyama scheme.

Usage examples:
    python simulation.py --broad_band 0.05
    python simulation.py --broad_band 0.5
    python simulation.py --broad_band 1.0

Optional:
    python simulation.py --broad_band 0.05 --force
    python simulation.py --broad_band 0.05 --lambda_min -50 --lambda_max 0 --n_lambdas 10
"""

import argparse
import os
import math
import numpy as np
from numba import njit


TWOPI = 2.0 * np.pi
KB_PN_NM_PER_K = 1.380649e-2  # pN nm / K
SQRT_2PI = math.sqrt(2.0 * math.pi)

# Save all outputs relative to the script location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

OBS_KEYS = (
    "occ_A",
    "occ_B",
    "occ_C",
    "rho_A",
    "rho_B",
    "power_B",
    "omega",
    "total_power",
    "sigma_B",
)
OBS_INDEX = {key: i for i, key in enumerate(OBS_KEYS)}
N_OBS = len(OBS_KEYS)


@njit(cache=True)
def periodic_diff(x, y, period):
    """Return x-y wrapped to (-period/2, period/2]."""
    d = x - y
    return d - period * np.floor(d / period + 0.5)


@njit(cache=True)
def periodic_gaussian_delta(x, center, sigma, period):
    """A narrow normalized periodic-Gaussian approximation to delta(x-center)."""
    d = periodic_diff(x, center, period)
    return np.exp(-0.5 * (d / sigma) ** 2) / (SQRT_2PI * sigma)


@njit(cache=True)
def periodic_gaussian_shape(x, center, sigma, period):
    """
    Area-normalized periodic Gaussian packet.

    The finite image sum below is more accurate than using only the shortest
    periodic distance, especially for broader packets.
    """
    d0 = periodic_diff(x, center, period)
    g = np.zeros_like(d0)

    # Images beyond +/-3 are negligible for sigma <= 1 on a 2pi-periodic circle.
    for n in range(-3, 4):
        d = d0 + n * period
        g += np.exp(-0.5 * (d / sigma) ** 2)

    return g / (SQRT_2PI * sigma)


@njit(cache=True)
def smooth_window(x, center, half_width, edge, period):
    """Smooth indicator of a periodic interval centered at center."""
    d = np.abs(periodic_diff(x, center, period))
    return 0.5 * (1.0 + np.tanh((half_width - d) / edge))


@njit(cache=True)
def sector_partition(theta, state_centers, period):
    """One-hot partition of the periodic angle into three non-overlapping sectors."""
    dA = np.abs(periodic_diff(theta, state_centers[0], period))
    dB = np.abs(periodic_diff(theta, state_centers[1], period))
    dC = np.abs(periodic_diff(theta, state_centers[2], period))

    maskA = (dA <= dB) & (dA <= dC)
    maskB = (~maskA) & (dB <= dC)
    maskC = ~(maskA | maskB)

    wA = np.zeros_like(theta)
    wB = np.zeros_like(theta)
    wC = np.zeros_like(theta)
    wA[maskA] = 1.0
    wB[maskB] = 1.0
    wC[maskC] = 1.0
    return wA, wB, wC


@njit(cache=True)
def motor_torque(theta, U0, F_drive):
    """Mechanical torque -dU/dtheta + F_drive for U(theta)=U0 cos(3 theta)."""
    return 3.0 * U0 * np.sin(3.0 * theta) + F_drive


@njit(cache=True)
def total_torque(theta, U0, F_drive, lam, z0, local_width, period):
    return motor_torque(theta, U0, F_drive) + lam * periodic_gaussian_shape(
        theta, z0, local_width, period
    )


@njit(cache=True)
def step_overdamped_inplace(
    theta,
    dtheta,
    noise,
    dt,
    mu,
    T_energy,
    U0,
    F_drive,
    lam,
    z0,
    local_width,
    period,
):
    """Advance all replicas by one overdamped Langevin step."""
    amp = math.sqrt(2.0 * mu * T_energy * dt)

    # Broadcast-style update for all replicas at once.
    tau = total_torque(theta, U0, F_drive, lam, z0, local_width, period)
    dtheta[:] = mu * tau * dt + amp * noise
    theta += dtheta


@njit(cache=True)
def accumulate_observables(
    theta,
    dtheta,
    sums,
    dt,
    F_drive,
    T_energy,
    U0,
    lam,
    z0,
    local_width,
    period,
    state_centers,
    state_half_width,
    state_edge,
    density_points,
    density_width,
    power_center,
    power_half_width,
    power_edge,
):
    """
    Accumulate per-trajectory time sums of stationary observables.

    State observables use the midpoint angle.  Current observables use the
    Stratonovich midpoint rule b(theta_mid) * dtheta / dt.
    """
    th_mid = theta - 0.5 * dtheta
    vel = dtheta / dt

    # Use a non-overlapping partition of the whole circle so that
    # tau_A + tau_B + tau_C = 1 exactly at every sampled midpoint.
    wA, wB, wC = sector_partition(th_mid, state_centers, period)

    rhoA = periodic_gaussian_delta(th_mid, density_points[0], density_width, period)
    rhoB = periodic_gaussian_delta(th_mid, density_points[1], density_width, period)

    # Local input-power flux in sector B.  The window is chosen away from z0,
    # so the current part satisfies the excluded-point condition b(z0)=0.
    w_power = smooth_window(th_mid, power_center, power_half_width, power_edge, period)
    power_B = F_drive * w_power * vel

    # Local entropy-production rate in the same sector-B window.  For this
    # overdamped model with constant temperature, the medium entropy flow is
    # torque * angular velocity / T_energy.
    tau_mid = total_torque(th_mid, U0, F_drive, lam, z0, local_width, period)
    sigma_B = w_power * tau_mid * vel / T_energy

    sums[0, :] += wA
    sums[1, :] += wB
    sums[2, :] += wC
    sums[3, :] += rhoA
    sums[4, :] += rhoB
    sums[5, :] += power_B
    sums[6, :] += vel
    sums[7, :] += F_drive * vel
    sums[8, :] += sigma_B


@njit(cache=True)
def simulate_steady_observables_numba(
    theta,
    lam,
    dt,
    mu,
    T_energy,
    U0,
    F_drive,
    z0,
    local_width,
    period,
    burn_steps,
    sample_steps,
    state_centers,
    state_half_width,
    state_edge,
    density_points,
    density_width,
    power_center,
    power_half_width,
    power_edge,
    seed,
):
    """Numba-compiled burn-in and sampling loops for one lambda value."""
    np.random.seed(seed)

    dtheta = np.empty_like(theta)
    noise = np.empty_like(theta)
    sums = np.zeros((N_OBS, theta.size), dtype=np.float64)

    for _ in range(burn_steps):
        noise[:] = np.random.standard_normal(theta.size)
        step_overdamped_inplace(
            theta,
            dtheta,
            noise,
            dt,
            mu,
            T_energy,
            U0,
            F_drive,
            lam,
            z0,
            local_width,
            period,
        )

    for _ in range(sample_steps):
        noise[:] = np.random.standard_normal(theta.size)
        step_overdamped_inplace(
            theta,
            dtheta,
            noise,
            dt,
            mu,
            T_energy,
            U0,
            F_drive,
            lam,
            z0,
            local_width,
            period,
        )
        accumulate_observables(
            theta,
            dtheta,
            sums,
            dt,
            F_drive,
            T_energy,
            U0,
            lam,
            z0,
            local_width,
            period,
            state_centers,
            state_half_width,
            state_edge,
            density_points,
            density_width,
            power_center,
            power_half_width,
            power_edge,
        )

    return sums


def default_lambdas():
    """Default lambda scan from the uploaded simulation script."""
    return np.linspace(-50.0, 0.0, 10)


def default_parameters(local_width):
    """Physically motivated F1-ATPase parameters."""
    T_kelvin = 298.0
    T_energy = KB_PN_NM_PER_K * T_kelvin
    return dict(
        period=TWOPI,
        mu=0.91,                  # rad / (s pN nm)
        T_energy=T_energy,        # pN nm, k_B T at 298 K
        U0=10.0 * T_energy,       # pN nm
        F_drive=120.0,            # pN nm, close to the giant-diffusion tilt
        z0=0.0,                   # local torque at a barrier of U0 cos(3 theta)
        local_width=local_width,  # rad
        dt=1.0e-5,                # s
        n_traj=50000,             # number of independent trajectories to run in parallel
        burn_steps=500000,        # number of initial steps to discard for equilibration
        sample_steps=500000,      # number of steps to accumulate observables for each lambda value
        # Minima of U(theta)=U0 cos(3 theta), used as coarse-grained states.
        state_centers=np.array([np.pi / 3.0, np.pi, 5.0 * np.pi / 3.0]),
        state_half_width=np.pi / 3.0,
        state_edge=0.06,
        # Kernel-density estimates of pi(theta) at state A and state B centers.
        density_points=np.array([np.pi / 3.0, np.pi]),
        density_width=0.06,
        # Local power window centered in state B, far from z0.
        power_center=np.pi,
        power_half_width=0.65,
        power_edge=0.06,
    )


def simulate_steady_observables(lam, params, seed=12345):
    """Return means and standard errors of all observables for one lambda."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, params["period"], size=params["n_traj"])

    sums = simulate_steady_observables_numba(
        np.array(theta0, dtype=np.float64, copy=True),
        lam,
        params["dt"],
        params["mu"],
        params["T_energy"],
        params["U0"],
        params["F_drive"],
        params["z0"],
        params["local_width"],
        params["period"],
        params["burn_steps"],
        params["sample_steps"],
        params["state_centers"],
        params["state_half_width"],
        params["state_edge"],
        params["density_points"],
        params["density_width"],
        params["power_center"],
        params["power_half_width"],
        params["power_edge"],
        seed,
    )

    traj_means = sums / params["sample_steps"]
    means = traj_means.mean(axis=1)
    sems = traj_means.std(axis=1, ddof=1) / math.sqrt(params["n_traj"])
    return means, sems


def scan_lambdas(lambdas, params, seed=20260421):
    """Compute steady-state observables over a list of local perturbation strengths."""
    lambdas = np.asarray(lambdas, dtype=np.float64)
    values = np.zeros((lambdas.size, len(OBS_KEYS)), dtype=np.float64)
    errors = np.zeros_like(values)

    for k, lam in enumerate(lambdas):
        means, sems = simulate_steady_observables(lam, params, seed=seed + 1009 * k)
        values[k] = means
        errors[k] = sems
        print(f"lambda={lam: .4g}  omega={means[OBS_INDEX['omega']]: .6g}  "
              f"occ_A={means[OBS_INDEX['occ_A']]: .6g}  "
              f"power_B={means[OBS_INDEX['power_B']]: .6g}")

    return dict(lambdas=lambdas, values=values, errors=errors, obs_keys=np.array(OBS_KEYS))


def save_results_npz(filename, results, params):
    """Save data with the same NPZ field names as the original combined script."""
    serial_params = {k: v for k, v in params.items() if np.isscalar(v)}
    np.savez(
        filename,
        lambdas=results["lambdas"],
        values=results["values"],
        errors=results["errors"],
        obs_keys=results["obs_keys"],
        state_centers=params["state_centers"],
        density_points=params["density_points"],
        **serial_params,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate steady-state mutual linearity data.")
    parser.add_argument("--broad_band", type=float, default=0.05,
                        help="Gaussian width / local_width. Used in the output filename prefix.")
    parser.add_argument("--lambda_min", type=float, default=-50.0)
    parser.add_argument("--lambda_max", type=float, default=0.0)
    parser.add_argument("--n_lambdas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--force", action="store_true",
                        help="Rerun simulation even if the NPZ file already exists.")
    parser.add_argument("--quick", action="store_true",
                        help="Use a small debug run for checking compilation.")
    return parser.parse_args()


def main():
    args = parse_args()
    file_suffix = f"{args.broad_band}_"
    results_file = os.path.join(DATA_DIR, file_suffix + "mutual_linearity.npz")

    params = default_parameters(local_width=args.broad_band)
    lambdas = np.linspace(args.lambda_min, args.lambda_max, args.n_lambdas)

    if args.quick:
        params["n_traj"] = 128
        params["burn_steps"] = 500
        params["sample_steps"] = 1500
        lambdas = np.linspace(-2.0, 2.0, 5)

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isfile(results_file) and not args.force:
        print(f"Found existing data file: {results_file}")
        print("Skip simulation. Use --force to overwrite it.")
        return

    print("Run simulation and save the results.")
    print(f"local_width={args.broad_band}, lambdas={lambdas}")
    results = scan_lambdas(lambdas, params, seed=args.seed)
    save_results_npz(results_file, results, params)
    print(f"Saved data to: {results_file}")


if __name__ == "__main__":
    main()
