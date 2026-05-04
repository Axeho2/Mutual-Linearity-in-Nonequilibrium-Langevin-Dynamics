"""
Non-stationary frequency-domain mutual linearity simulation for the F1-ATPase
 tilted-periodic-potential model.

This file only runs simulations and saves data.  The sampling method and the
saved npz format match the original combined script.

Model on the unwrapped angular coordinate theta:

    d theta = mu * [3 U0 sin(3 theta) + F_drive
                    + lam * g_l(theta - z0)] dt
              + sqrt(2 mu T) dW_t .

Here g_l is a periodic Gaussian shape function with fixed peak height and width
set by local_width.  The perturbation strength lambda is scanned, and the same
initial distribution is used for all lambda values.  For each trajectory we
accumulate the Laplace transform of time-dependent state-current observables,

    \hat Q_lambda(omega) = int_0^infty exp(-omega t) <Q_lambda(t)> dt,

at the default omega values.  The Laplace integral is directly truncated at
T_cut = 5 s.  No stationary-tail correction is added.

No classes are used.  The stepping routine evolves all replicas in one array.
The expensive loops are numba-compiled.
"""

import argparse
import os
import math
import numpy as np
from numba import njit

# 0.04, 0.5, 1.0
band_width = 0.1
DEFAULT_LOCAL_WIDTH = band_width
DEFAULT_LAMBDAS = np.linspace(-150.0, 150.0, 10)
DEFAULT_OMEGAS = np.array([2.0, 5.0, 10.0], dtype=np.float64)

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
    """Peak-normalized periodic Gaussian shape: max value = 1 at x=center."""
    d = periodic_diff(x, center, period)
    return np.exp(-0.5 * (d / sigma) ** 2)


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
    tau = total_torque(theta, U0, F_drive, lam, z0, local_width, period)
    dtheta[:] = mu * tau * dt + amp * noise
    theta += dtheta


@njit(cache=True)
def accumulate_laplace_observables(
    theta,
    dtheta,
    laplace_sums,
    weights,
    dt,
    F_drive,
    T_energy,
    U0,
    lam,
    z0,
    local_width,
    period,
    state_centers,
    density_points,
    density_width,
    power_center,
    power_half_width,
    power_edge,
):
    """
    Accumulate per-trajectory Laplace sums of time-dependent observables.

    The instantaneous observables are evaluated at the Stratonovich midpoint.
    Each omega uses its own scalar weight exp(-omega t_mid) * dt.
    """
    th_mid = theta - 0.5 * dtheta
    vel = dtheta / dt

    wA, wB, wC = sector_partition(th_mid, state_centers, period)
    rhoA = periodic_gaussian_delta(th_mid, density_points[0], density_width, period)
    rhoB = periodic_gaussian_delta(th_mid, density_points[1], density_width, period)

    w_power = smooth_window(th_mid, power_center, power_half_width, power_edge, period)
    power_B = F_drive * w_power * vel
    total_power = F_drive * vel

    # Local entropy-production rate in the same sector-B window.  For this
    # overdamped model with constant temperature, the medium entropy flow is
    # torque * angular velocity / T_energy.
    tau_mid = total_torque(th_mid, U0, F_drive, lam, z0, local_width, period)
    sigma_B = w_power * tau_mid * vel / T_energy

    for io in range(weights.size):
        wdt = weights[io] * dt
        laplace_sums[io, 0, :] += wdt * wA
        laplace_sums[io, 1, :] += wdt * wB
        laplace_sums[io, 2, :] += wdt * wC
        laplace_sums[io, 3, :] += wdt * rhoA
        laplace_sums[io, 4, :] += wdt * rhoB
        laplace_sums[io, 5, :] += wdt * power_B
        laplace_sums[io, 6, :] += wdt * vel
        laplace_sums[io, 7, :] += wdt * total_power
        laplace_sums[io, 8, :] += wdt * sigma_B


@njit(cache=True)
def simulate_laplace_observables_numba(
    theta,
    lam,
    omegas,
    dt,
    mu,
    T_energy,
    U0,
    F_drive,
    z0,
    local_width,
    period,
    total_steps,
    state_centers,
    density_points,
    density_width,
    power_center,
    power_half_width,
    power_edge,
    seed,
):
    """Numba-compiled non-stationary sampling loop for one lambda value."""
    np.random.seed(seed)

    n_omega = omegas.size
    n_traj = theta.size
    dtheta = np.empty_like(theta)
    noise = np.empty_like(theta)

    laplace_sums = np.zeros((n_omega, N_OBS, n_traj), dtype=np.float64)

    # Midpoint weights.  At step k, midpoint time is (k+0.5)*dt.
    weights = np.empty(n_omega, dtype=np.float64)
    decay = np.empty(n_omega, dtype=np.float64)
    for io in range(n_omega):
        weights[io] = math.exp(-omegas[io] * 0.5 * dt)
        decay[io] = math.exp(-omegas[io] * dt)

    for _ in range(total_steps):
        noise[:] = np.random.standard_normal(n_traj)
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
        accumulate_laplace_observables(
            theta,
            dtheta,
            laplace_sums,
            weights,
            dt,
            F_drive,
            T_energy,
            U0,
            lam,
            z0,
            local_width,
            period,
            state_centers,
            density_points,
            density_width,
            power_center,
            power_half_width,
            power_edge,
        )
        for io in range(n_omega):
            weights[io] *= decay[io]

    # Direct finite-time truncation: no stationary-tail correction is added.
    return laplace_sums


def default_parameters(local_width=DEFAULT_LOCAL_WIDTH):
    """Physically motivated F1-ATPase parameters and non-stationary run settings."""
    T_kelvin = 298.0
    T_energy = KB_PN_NM_PER_K * T_kelvin
    return dict(
        period=TWOPI,
        mu=0.91,                  # rad / (s pN nm)
        T_energy=T_energy,        # pN nm, k_B T at 298 K
        U0=10.0 * T_energy,       # pN nm
        F_drive=120.0,            # pN nm, close to the giant-diffusion tilt
        z0=0.0,                   # local torque at a barrier of U0 cos(3 theta)
        local_width=local_width,
        dt=1.0e-5,                # s
        n_traj=50000,             # independent trajectories evolved in parallel
        total_steps=500000,       # 5 s at dt=1e-5; direct finite-time truncation
        tail_average_steps=0,     # kept for backward compatibility; no tail correction
        # Initial distribution, independent of lambda.  This creates a clear
        # relaxation signal before the system reaches the lambda-dependent NESS.
        initial_center=np.pi / 3.0,
        initial_width=0.08,
        # Minima of U(theta)=U0 cos(3 theta), used as coarse-grained states.
        state_centers=np.array([np.pi / 3.0, np.pi, 5.0 * np.pi / 3.0]),
        # Kernel-density estimates of pi(theta,t) at state A and state B centers.
        density_points=np.array([np.pi / 3.0, np.pi]),
        density_width=0.06,
        # Local power window centered in state B, far from z0.
        power_center=np.pi,
        power_half_width=0.65,
        power_edge=0.06,
    )


def sample_initial_condition(params, seed):
    """Return the same kind of lambda-independent initial distribution for every scan point."""
    rng = np.random.default_rng(seed)
    theta0 = params["initial_center"] + params["initial_width"] * rng.standard_normal(params["n_traj"])
    return np.asarray(theta0, dtype=np.float64)


def simulate_laplace_observables(lam, omegas, params, seed=12345):
    """Return means and standard errors of Laplace-transformed observables."""
    theta0 = sample_initial_condition(params, seed)
    laplace_sums = simulate_laplace_observables_numba(
        np.array(theta0, dtype=np.float64, copy=True),
        lam,
        np.asarray(omegas, dtype=np.float64),
        params["dt"],
        params["mu"],
        params["T_energy"],
        params["U0"],
        params["F_drive"],
        params["z0"],
        params["local_width"],
        params["period"],
        params["total_steps"],
        params["state_centers"],
        params["density_points"],
        params["density_width"],
        params["power_center"],
        params["power_half_width"],
        params["power_edge"],
        seed,
    )

    means = laplace_sums.mean(axis=2)
    sems = laplace_sums.std(axis=2, ddof=1) / math.sqrt(params["n_traj"])
    return means, sems


def scan_lambdas(lambdas, omegas, params, seed=20260429):
    """Compute Laplace-domain observables over a list of local perturbation strengths."""
    lambdas = np.asarray(lambdas, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    values = np.zeros((lambdas.size, omegas.size, len(OBS_KEYS)), dtype=np.float64)
    errors = np.zeros_like(values)

    for k, lam in enumerate(lambdas):
        means, sems = simulate_laplace_observables(lam, omegas, params, seed=seed + 1009 * k)
        values[k] = means
        errors[k] = sems
        msg = f"lambda={lam: .4g}"
        for io, om in enumerate(omegas):
            msg += (f"  omega={om:.2f}: "
                    f"hat_occ_A={means[io, OBS_INDEX['occ_A']]: .6g} "
                    f"hat_power_B={means[io, OBS_INDEX['power_B']]: .6g}")
        print(msg)

    return dict(
        lambdas=lambdas,
        omegas=omegas,
        values=values,
        errors=errors,
        obs_keys=np.array(OBS_KEYS),
    )


def save_results_npz(filename, results, params):
    """Save sampled data and scalar parameters for later reuse."""
    serial_params = {k: v for k, v in params.items() if np.isscalar(v)}
    np.savez(
        filename,
        lambdas=results["lambdas"],
        omegas=results["omegas"],
        values=results["values"],
        errors=results["errors"],
        obs_keys=results["obs_keys"],
        state_centers=params["state_centers"],
        density_points=params["density_points"],
        **serial_params,
    )


def results_filename(local_width):
    file_suffix = f"nonstationary_{local_width}_"
    return os.path.join(DATA_DIR, file_suffix + "laplace_mutual_linearity.npz")


def quick_debug_run():
    """Small run for checking compilation, not for production figures."""
    params = default_parameters()
    params["n_traj"] = 256
    params["total_steps"] = 5000
    params["tail_average_steps"] = 0
    lambdas = np.linspace(-2.0, 2.0, 5)
    omegas = DEFAULT_OMEGAS
    results = scan_lambdas(lambdas, omegas, params, seed=7)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run non-stationary Laplace-domain mutual-linearity simulations.")
    parser.add_argument("--band_width", type=float, default=band_width, help="Gaussian perturbation width.")
    parser.add_argument("--force", action="store_true", help="Re-run simulation even if the cached data file exists.")
    parser.add_argument("--debug", action="store_true", help="Run a small debug simulation and do not save production data.")
    args = parser.parse_args()

    if args.debug:
        quick_debug_run()
        return

    params = default_parameters(local_width=args.band_width)
    lambdas = DEFAULT_LAMBDAS
    omegas = DEFAULT_OMEGAS
    filename = results_filename(args.band_width)

    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isfile(filename) and not args.force:
        print(f"Found existing data file: {filename}")
        print("Skip simulation. Use --force to regenerate it.")
        return

    print("Run non-stationary Laplace-domain simulation and save the results.")
    results = scan_lambdas(lambdas, omegas, params, seed=20260429)
    save_results_npz(filename, results, params)
    print(f"Saved data to {filename}")


if __name__ == "__main__":
    main()
