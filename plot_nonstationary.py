"""
Plot non-stationary frequency-domain mutual linearity for the F1-ATPase model.

This file only reads saved npz data and makes figures.  The plotting style and
figure outputs match the original combined script.  No abnormal-point handling
is applied.
"""

import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Keep defaults consistent with simulation_nonstationary.py.
band_width = 0.1
file_suffix = f"nonstationary_{band_width}_"

SINGLE_COLUMN_WIDTH_IN = 1.95
FIG_HEIGHT_IN = 1.55

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
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


def results_filename(local_width):
    suffix = f"nonstationary_{local_width}_"
    return os.path.join(DATA_DIR, suffix + "laplace_mutual_linearity.npz")


def output_prefix(local_width):
    return f"nonstationary_{local_width}_"


def load_results_npz(filename):
    data = np.load(filename, allow_pickle=True)
    return dict(
        lambdas=data["lambdas"],
        omegas=data["omegas"],
        values=data["values"],
        errors=data["errors"],
        obs_keys=data["obs_keys"],
    )


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def get_obs(results, key, omega_index):
    idx = OBS_INDEX[key]
    return results["values"][:, omega_index, idx], results["errors"][:, omega_index, idx]


def fit_line(x, y, xerr=None, yerr=None):
    """
    Symmetric line fit using Deming regression.
    If xerr/yerr are not provided, fall back to lambda_ratio = 1
    (orthogonal regression / total least squares).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xbar = np.mean(x)
    ybar = np.mean(y)
    dx = x - xbar
    dy = y - ybar

    Sxx = np.mean(dx * dx)
    Syy = np.mean(dy * dy)
    Sxy = np.mean(dx * dy)

    if abs(Sxy) < 1.0e-300:
        slope = np.nan
        intercept = np.nan
        r2 = np.nan
        cov = np.full((2, 2), np.nan)
        return slope, intercept, cov, r2

    if xerr is None or yerr is None:
        lambda_ratio = 1.0
    else:
        sx2 = np.mean(np.asarray(xerr, dtype=np.float64) ** 2)
        sy2 = np.mean(np.asarray(yerr, dtype=np.float64) ** 2)
        lambda_ratio = sy2 / sx2 if sx2 > 0 else 1.0

    disc = (Syy - lambda_ratio * Sxx) ** 2 + 4.0 * lambda_ratio * Sxy ** 2
    slope = (Syy - lambda_ratio * Sxx + np.sqrt(disc)) / (2.0 * Sxy)
    intercept = ybar - slope * xbar

    r = np.corrcoef(x, y)[0, 1]
    r2 = r * r
    cov = np.full((2, 2), np.nan)
    return slope, intercept, cov, r2


def save_figure_both(fig, filebase):
    """Save the same figure as both PDF and SVG."""
    fig.savefig(filebase + ".pdf")
    fig.savefig(filebase + ".svg")


def plot_laplace_mutual_linearity(
    results,
    xkey,
    ykey,
    xlabel,
    ylabel,
    filebase,
    panel_label=None,
    annotate=True,
):
    """Plot one mutual-linearity panel with all omega values on the same axes."""
    set_style()
    omegas = results["omegas"]
    colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(
        1, 1,
        figsize=(SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    stats = {}
    for io, omega in enumerate(omegas):
        x, xerr = get_obs(results, xkey, io)
        y, yerr = get_obs(results, ykey, io)
        slope, intercept, cov, r2 = fit_line(x, y, xerr, yerr)
        stats[f"omega_{omega:.3g}"] = (slope, intercept, r2)

        color = colors[io % len(colors)]
        marker = markers[io % len(markers)]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=color,
            ecolor="0.55",
            elinewidth=0.70,
            capsize=1.8,
            markersize=3,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            label=rf"$\omega={omega:.1f}$",
            zorder=3,
        )

        if np.isfinite(slope):
            xs = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(xs, slope * xs + intercept, color=color, lw=1.15, zorder=2)

        if annotate:
            text = rf"$R^2_{{{omega:.1f}}}={r2:.4f}$"
            ax.text(
                0.97,
                0.06 + 0.13 * io,
                text,
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                color=color,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", top=True, right=True)
    if panel_label is not None:
        ax.text(0.03, 0.96, panel_label, transform=ax.transAxes, va="top", ha="left")
    ax.legend(frameon=False, handlelength=1.3, loc="best")

    save_figure_both(fig, filebase)
    plt.close(fig)
    return stats


def plot_laplace_observables_vs_lambda(results, filebase):
    """Optional diagnostic: selected Laplace observables versus lambda for each omega."""
    set_style()
    lambdas = results["lambdas"]
    omegas = results["omegas"]
    colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]

    fig, ax = plt.subplots(
        1, 1,
        figsize=(SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    for io, omega in enumerate(omegas):
        y, yerr = get_obs(results, "occ_A", io)
        ax.errorbar(
            lambdas,
            y,
            yerr=yerr,
            fmt="o-",
            color=colors[io % len(colors)],
            ecolor="0.55",
            elinewidth=0.70,
            capsize=1.8,
            markersize=3,
            markerfacecolor="white",
            markeredgewidth=0.85,
            label=rf"$\omega={omega:.1f}$",
        )

    ax.set_xlabel(r"torque strength $\lambda$")
    ax.set_ylabel(r"$\widehat{\tau}_A(\omega)$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, handlelength=1.5)
    save_figure_both(fig, filebase)
    plt.close(fig)


def make_all_figures(results, outdir="figures", prefix=file_suffix):
    os.makedirs(outdir, exist_ok=True)

    stats = {}
    stats["occA_occB"] = plot_laplace_mutual_linearity(
        results,
        "occ_B",
        "occ_A",
        r"$\widehat{\tau}_B(\omega)$",
        r"$\widehat{\tau}_A(\omega)$",
        os.path.join(outdir, prefix + "hat_occA_vs_hat_occB"),
        panel_label=None,
    )
    stats["rhoA_rhoB"] = plot_laplace_mutual_linearity(
        results,
        "rho_B",
        "rho_A",
        r"$\widehat{\pi}_{\rho}(\theta_B,\omega)$",
        r"$\widehat{\pi}_{\rho}(\theta_A,\omega)$",
        os.path.join(outdir, prefix + "hat_rhoA_vs_hat_rhoB"),
        panel_label=None,
    )
    stats["powerB_occC"] = plot_laplace_mutual_linearity(
        results,
        "occ_C",
        "power_B",
        r"$\widehat{\tau}_C(\omega)$",
        r"$\widehat{P}^{\rm in}_B(\omega)$",
        os.path.join(outdir, prefix + "hat_powerB_vs_hat_occC"),
        panel_label=None,
    )
    stats["sigmaB_occC"] = plot_laplace_mutual_linearity(
        results,
        "occ_C",
        "sigma_B",
        r"$\widehat{\tau}_C(\omega)$",
        r"$\widehat{\sigma}_B(\omega)$",
        os.path.join(outdir, prefix + "hat_sigmaB_vs_hat_occC"),
        panel_label=None,
    )
    plot_laplace_observables_vs_lambda(
        results,
        os.path.join(outdir, prefix + "hat_occA_vs_lambda"),
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Plot non-stationary Laplace-domain mutual-linearity data.")
    parser.add_argument("--band_width", type=float, default=band_width, help="Gaussian perturbation width used in saved data.")
    parser.add_argument("--allow_missing", action="store_true", help="Do not raise an error if the requested data file is missing.")
    args = parser.parse_args()

    filename = results_filename(args.band_width)
    if not os.path.isfile(filename):
        msg = f"Data file not found: {filename}"
        if args.allow_missing:
            print(msg)
            return
        raise FileNotFoundError(msg)

    results = load_results_npz(filename)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    stats = make_all_figures(results, outdir=FIGURES_DIR, prefix=output_prefix(args.band_width))
    print(stats)


if __name__ == "__main__":
    main()
