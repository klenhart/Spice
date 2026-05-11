from typing import Tuple, Dict, List, Optional, Set, Iterable, Literal, Union, Any, Sequence
import gzip
import re
import itertools
from collections import defaultdict
from pathlib import Path
from scipy import stats
import warnings
import csv
import json
import os
import itertools
from itertools import combinations
import math 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
################## PLOTTING HELPER FUNCTIONS ####################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _clip01(x: np.ndarray) -> np.ndarray:
    """
    Helper function helping with clipping RMSD values at boundaries.
    """
    return np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12)


def _cdf_ppf_from_row(row):
    """
    Return CDF and PPF from the fit. Will be used for QQ-plots
    of the best distribution fit. Wroks on rows of the dataframe
    containing parameters of fitted distribution functions.
    """
    model = row["model"]
    # parse model params from string
    ps = row["params"]
    if model == "beta":
        # extract parameters
        a = float(ps.split(",")[0].split("=")[1]); b = float(ps.split(",")[1].split("=")[1])
        CDF = lambda z: stats.beta.cdf(z, a, b, loc=0, scale=1)
        PPF = lambda p: stats.beta.ppf(p, a, b, loc=0, scale=1)
        return CDF, PPF

    if model == "johnsonsb":
        a = float(ps.split(",")[0].split("=")[1]); b = float(ps.split(",")[1].split("=")[1])
        CDF = lambda z: stats.johnsonsb.cdf(z, a, b, loc=0, scale=1)
        PPF = lambda p: stats.johnsonsb.ppf(p, a, b, loc=0, scale=1)
        return CDF, PPF

    if model == "powerlaw":
        a = float(ps.split("=")[1])
        CDF = lambda z: stats.powerlaw.cdf(z, a, loc=0, scale=1)
        PPF = lambda p: stats.powerlaw.ppf(p, a, loc=0, scale=1)
        return CDF, PPF

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
######################## PLOTTING FUNCTIONS #####################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def parse_params(param_str):
    """
    Works on rows of the spice.winners_by_AIC output table and 
    reads fitted beta parameters.
    Returns a and b parameters as floats.

    Notes: This function also expects that beta is the best fit.
    """
    parts = dict(s.strip().split("=") for s in param_str.replace(" ", "").split(","))
    return float(parts["a"]), float(parts["b"])
    # m = re.search(r"a\s*=\s*([0-9.]+)\s*,\s*b\s*=\s*([0-9.]+)", str(param_str))
    # if not m:
    #     raise ValueError(f"Could not parse params: {param_str}")
    # return float(m.group(1)), float(m.group(2))

def ecdf_vals(x):
    """
    Returns sorted RMSD (x, sorted) and corresponding
    ECDF (y) values from the empirical discrete RMSD distribution
    of AT-level references.
    """
    x = np.asarray(x, float)
    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y

def plot_ecdf_with_beta_cdf(pooled_by_AT,
                            winner_fits_df,
                            keys="all",
                            ncols=3,
                            figsize_per_row=3.4,
                            grid_points=600,
                            clip_eps=1e-12):
    """
    Plot ECDFs of empirical samples and overlay Beta CDF fits.

    Input:
        pooled_by_AT: pooled_by_AT output of spice.merge_bins
        winner_fits_df: DataFrame containing the fitted beta distribution
                        parameters, output of spice.winners_by_AIC
        keys: "all" or list of AT bin ids to display.
        ncols: Number of columns in the final plot
        grid_points: Number of grid_points
        clip_eps: To clip RMSD values at boundaries

    Notes:
        This function expects beta parameters. If the best fit was another
        distribution function, you need to modify this.
    """

    keys = sorted(pooled_by_AT.keys()) if keys == "all" else sorted(keys)
    n = len(keys)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5, nrows * figsize_per_row),
        sharex=False, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, keys):
        x = np.asarray(pooled_by_AT[k], float)

        # ECDF
        xs, ys = ecdf_vals(x)
        ax.step(xs, ys, where="post", linewidth=2, label="ECDF")
        xmin, xmax = float(xs[0]), float(xs[-1])
        pad = 0.02 * (xmax - xmin if xmax > xmin else 1.0)
        xlo = max(0.0, xmin - pad)
        xhi = min(1.0, xmax + pad)
        ax.set_xlim(xlo, xhi)

        xx = np.linspace(max(clip_eps, xlo), min(1 - clip_eps, xhi), grid_points)
        # select the subset of a dataframe if not all AT bins should be displayed
        df_use = winner_fits_df.loc[winner_fits_df["AT"] == k] if keys != "all" else winner_fits_df

        # fitted Beta CDFs
        for _, row in df_use.iterrows():
            a, b = parse_params(row["params"])
            ax.plot(xx, stats.beta.cdf(xx, a, b), linewidth=2, linestyle="--",
                    label=f"Beta CDF (a={a:.3g}, b={b:.3g})")

        ax.set_title(f"AT = {k}")
        ax.set_ylim(0, 1)

        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True)

    for ax in axes[len(keys):]:
        ax.axis("off")

    fig.supylabel("Cumulative probability")
    fig.supxlabel("RMSD")
    # fig.suptitle("Empirical ECDF with Beta CDF fit", y=1.02)
    plt.tight_layout()
    plt.show()



def plot_all_bins_QQ(pooled_by_AT: Dict[int, List[float]],
                     fit_df: pd.DataFrame,
                     model_choice: str = "best",
                     winner_by_AIC: Optional[pd.DataFrame] = None,
                     ncols: int = 3):
    """
    QQ-plots of empirical and fitted quantiles. Creates one QQ subplot per bin.
    If model_choice == "best", uses the AIC winner for that bin.
    If model_choice == "best", then the used needs to provide the winner_by_AIC
    dataframe obtained from spice.winners_by_AIC. If not "best",
    uses the specified model for every bin.
    
    Input:
        pooled_by_AT: output of either spice.calc_obs_metrics (grouped_by_AT)
                      or spice.merge_bins (summary_dict)
        fit_df: Dataframe of all fits obtained from spice.fit_distributions_scipy
        model_coice: Default: "best" -> requires winner_by_AIC. Possible values:
                                "beta","johnsonsb", "powerlaw".
        winner_by_AIC: Dataframe output of spice.winners_by_AIC. Only provide if
                       model_choice == "best"
        ncols: Number of columns to display in the plot. Default: 3

    """
    if model_choice == "best":
        chosen = winner_by_AIC
    else:
        chosen = (fit_df[fit_df["model"] == model_choice]
                  .sort_values("AT")
                  .reset_index(drop=True))

    ATs = chosen["AT"].tolist()
    n_bins = len(ATs)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
    axes = axes.ravel()

    for i, (_, row) in enumerate(chosen.iterrows()):
        AT = row["AT"]
        x = _clip01(np.asarray(pooled_by_AT[AT], float))
        x = x[np.isfinite(x)]
        n = x.size
        x_sorted = np.sort(x)
        idx = np.arange(n)
        p = (idx + 0.5) / n
        x_emp = x_sorted[idx]

        CDF, PPF = _cdf_ppf_from_row(row)
        # model quantiles 
        q_theory = PPF(p)
        # empirical quantiles
        x_emp    = x_sorted[idx]
        ax = axes[i]
        ax.scatter(q_theory, x_emp, s=6, alpha=0.8)

        # Diagonal reference line in data units
        lo = np.nanmin([np.nanmin(q_theory), np.nanmin(x_emp)])
        hi = np.nanmax([np.nanmax(q_theory), np.nanmax(x_emp)])
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="black")

        title_model = row["model"]
        ax.set_title(f"AT={AT} — {title_model}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Theoretical quantiles (model)")
        ax.set_ylabel("Empirical quantiles (data)")

    # hide any empty panels
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("QQ plots — all bins", y=0.995, fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_distributions_per_bin(grouped_by_AT: Dict[int, Dict[str, List[float]]],
                               metric: str = "rmsd",
                               bins: int = 50,
                               max_bins_to_plot: int = 9,
                               logscale: bool = True):
    """
    Plots AT-level histograms of RMSD values when submitting a grouped_by_AT/summary_dict
    dictionary output from spice.calc_obs_metrics or spice.build_AT_reference_dist.

    Parameters:
        grouped_by_AT: grouped_by_AT/summary_dict output from spice.calc_obs_metrics
                       or spice.build_AT_reference_dist.
        metric: Denotes the key within grouped_by_AT[AT] dict that stores values. Default: "rmsd"
        bins: Number of histogram bins.
        max_bins_to_plot: Maximum number of AT-bins to display.
    """
    AT_bins = sorted(grouped_by_AT.keys())[:max_bins_to_plot]
    num_bins = len(AT_bins)
    n_cols = 3
    n_rows = (num_bins + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axs = axs.flatten()

    for i, meta_bin in enumerate(AT_bins):
        data = grouped_by_AT[meta_bin].get(metric, [])
        sns.histplot(data, bins=bins, kde=False, ax=axs[i], color="skyblue", edgecolor="black", stat="density")
        axs[i].set_title(f"Bin {meta_bin}")
        axs[i].set_xlabel(metric.upper())
        axs[i].set_ylabel("Count")
        if logscale:
            axs[i].set_yscale("log")
        axs[i].grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle(f"Distributions of {metric.upper()} values across AT-Bins", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_hits_bar(genes_by_txcount, hits_by_bin, sort_bins=True, title=None):
    """
    Plots bar plot of hits obtained from find_two_cluster_genes_by_txcount_active_only.
    """
    bins = [b for b, hits in hits_by_bin.items() if hits]
    if sort_bins:
        bins = sorted(bins)

    if not bins:
        print("No bins with hits to plot.")
        return

    totals = [len(genes_by_txcount.get(b, [])) for b in bins]
    hits   = [len(hits_by_bin.get(b, []))      for b in bins]
    props = [(h / t) if t else 0.0 for h, t in zip(hits, totals)]
    x = list(range(len(bins)))

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(bins)), 5))
    ax.bar(x, props)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bins], rotation=0, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of genes")
    ax.set_xlabel("AT-bin")
    ax.legend()

    for i, (h, t, p) in enumerate(zip(hits, totals, props)):
        ax.text(
            i,
            p,
            f"{h}/{t}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_half_gap_compare_multi(per_gap_dfs: Sequence[pd.DataFrame],
                                titles: Sequence[str],
                                ncols: int = 2,
                                figsize: Optional[Tuple[float, float]] = None,
                                suptitle: Optional[str] = "Median EMD by gap: lower vs upper sets (95% CI)"):
    """
    Plot lower vs upper median EMD by gap with 95% CI for multiple per_gap_df inputs obtained from
    half_gap_compare.
    """
    n_plots = len(per_gap_dfs)
    n_total = n_plots + 1
    nrows = math.ceil(n_total / ncols)

    if figsize is None:
        figsize = (6.5 * ncols, 4.8 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    legend_handles = None
    legend_labels = None

    # ---- main subplots ----
    for idx, (df, ax) in enumerate(zip(per_gap_dfs, axes.ravel())):
        d = df.sort_values("gap")
        x = d["gap"].to_numpy()
        y_lo = d["lower_median"].to_numpy(dtype=float)
        yerr_lo = np.vstack([
            y_lo - d["lower_ci_low"].to_numpy(dtype=float),
            d["lower_ci_high"].to_numpy(dtype=float) - y_lo,
        ])

        y_up = d["upper_median"].to_numpy(dtype=float)
        yerr_up = np.vstack([
            y_up - d["upper_ci_low"].to_numpy(dtype=float),
            d["upper_ci_high"].to_numpy(dtype=float) - y_up,
        ])

        h1 = ax.errorbar(x, y_lo, yerr=yerr_lo, fmt="o-", capsize=3,
                         label="Lower set (median EMD)")
        h2 = ax.errorbar(x, y_up, yerr=yerr_up, fmt="s--", capsize=3,
                         label="Upper set (median EMD)")

        if legend_handles is None:
            legend_handles = [h1, h2]
            legend_labels = ["Lower set: [2,3,4,5]", "Upper set: [7,8,9,10]"]

        ax.set_xticks(range(1,4))
        ax.set_title(titles[idx])
        ax.set_xlabel("Bin gap |i-j|")
        ax.set_ylabel("Median EMD per gap")
        ax.grid(True)

    legend_ax = axes.ravel()[n_plots]
    legend_ax.axis("off")

    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=True,
        fontsize=plt.rcParams["font.size"],
    )
    for j in range(n_plots + 1, nrows * ncols):
        axes.ravel()[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    plt.show()


def heatmap_EMD_dists(comparison_dfs: pd.DataFrame,
                      cbar_label: str = "EMD",
                      titles: Optional[Sequence[str]] = None,
                      bins_order: Optional[Sequence] = None,
                      ncols: Optional[int] = None,
                      cmap: str = "viridis",
                      rotate_xticks: int = 45,
                      figsize: Optional[tuple[float, float]] = None):

    n = len(comparison_dfs)

    if bins_order is None:
        bins_order = list(comparison_dfs[0].index)

    mats = []
    vmin, vmax = np.inf, -np.inf
    for df in comparison_dfs:
        mat = df.reindex(index=bins_order, columns=bins_order).to_numpy(dtype=float)
        mats.append(mat)
        vmin = min(vmin, np.nanmin(mat))
        vmax = max(vmax, np.nanmax(mat))

    if ncols is None:
        ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    if figsize is None:
        figsize = (4.2 * ncols + 0.9, 3.8 * nrows)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=[1]*ncols + [0.06])

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))

    cax = fig.add_subplot(gs[:, -1])

    last_im = None
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        last_im = ax.imshow(
            mats[i],
            origin="upper",
            interpolation="nearest",
            aspect="equal",     
            vmin=vmin, vmax=vmax,
            cmap=cmap
        )

        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

        ax.set_xticks(range(len(bins_order)))
        ax.set_yticks(range(len(bins_order)))
        ax.set_xticklabels(bins_order, rotation=rotate_xticks, ha="right")
        ax.set_yticklabels(bins_order)

        if titles is not None:
            ax.set_title(titles[i])

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(cbar_label)

    plt.show()



def plot_distributions_from_result(result_table: pd.DataFrame,
                                   metric: str = "RMSD",
                                   bins: int = 50,
                                   max_bins_to_plot: int = 9,
                                   logscale: bool = True,
                                   highlight_genes: Optional[List[str]] = None,
                                   gene_col: str = "GeneID"):
    """
    Plots histograms of metric distributions for each AT-bin and optionally
    highlights selected genes within each bin.

    Highlighted genes that share the same metric value are collapsed into
    a single vertical line, with the number of genes shown above that line.

    Spice compatibility functions:
        build_results_table

    Input
        result_table : Input table containing AT bins, metric values and gene names. 
                        Output of Spice function "build_results_table".
        metric : Metric to plot on the x-axis. Default: "RMSD"
        bins : Histogram bins. Default: 50
        max_bins_to_plot : Maximum number of AT bins to display. Default: 9
        logscale :  Whether to use log scale on the y-axis. Default:True
        highlight_genes : List of genes to highlight in the corresponding AT-bin distributions.
        gene_col : Name of the column containing gene ID. Default: "GeneID"
    """
    if highlight_genes is None:
        highlight_genes = []

    required_cols = {"AT", metric}
    missing = required_cols - set(result_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if highlight_genes and gene_col not in result_table.columns:
        raise ValueError(
            f"To use `highlight_genes`, `result_table` must contain a '{gene_col}' column."
        )

    AT_bins = sorted(result_table["AT"].dropna().unique())[:max_bins_to_plot]

    num_bins = len(AT_bins)
    n_cols = 3
    n_rows = (num_bins + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axs = np.array(axs).reshape(-1)

    for i, at_bin in enumerate(AT_bins):
        ax = axs[i]
        bin_df = result_table[result_table["AT"] == at_bin]
        data = bin_df[metric].dropna()

        sns.histplot(
            data,
            bins=bins,
            kde=False,
            ax=ax,
            color="skyblue",
            edgecolor="black",
            stat="density",
        )

        ax.set_title(f"Bin {at_bin} (n={len(data)})")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Density")

        if logscale:
            ax.set_yscale("log")

        ax.grid(True)

        # Highlight selected genes
        if highlight_genes:
            highlight_df = bin_df[bin_df[gene_col].isin(highlight_genes)].copy()
            highlight_df = highlight_df.dropna(subset=[metric])

            if not highlight_df.empty:
                value_counts = (
                    highlight_df.groupby(metric)[gene_col]
                    .nunique()
                    .sort_index()
                )

                ymin, ymax = ax.get_ylim()

                if logscale:
                    text_y = ymax / 1.3
                else:
                    text_y = ymax * 0.95

                for x, count in value_counts.items():
                    xmin, xmax = ax.get_xlim()
                    x_offset = (xmax - xmin) * 0.01
                    ax.axvline(
                        x=x,
                        color="red",
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.9,
                    )

                    ax.text(
                        x - x_offset,
                        text_y,
                        str(count),
                        color="red",
                        fontsize=9,
                        fontweight="bold",
                        ha="right",
                        va="bottom" if not logscale else "top",
                        clip_on=False,
                    )
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_boot(ax, stats, title, ylim=None):
    """ Plotting function for output of bootstrap_by_bin"""
    x = stats["xpos"]
    # median
    # ax.plot(x, stats["med_mean"], "-o", label="Median (bootstrap mean)")
    # ax.fill_between(x, stats["med_lo"], stats["med_hi"], alpha=0.25, label="Median 95% CI")
    # max
    ax.plot(x, stats["max_mean"], "-o", label="Average maximum across subsamples")
    ax.fill_between(x, stats["max_lo"], stats["max_hi"], alpha=0.25, label="Max 95% CI")

    ax.set_xticks(x, stats["labels"])
    ax.set_xlabel("Transcript Number")
    ax.set_ylabel("RMSD")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
##### Showing the RMSD bias towards high transcript counts #####
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def subsample_by_bin(df, group_col, value_col, bins, size_per_bin, n_boot, rng, ci):
    """
    Subsamples each bin to the smallest bin-size specified by size_per_bin and obtains
    maximum and median values (values in df denoted by value_col) across subsampling
    iterations. The function then aggregates the maximum and median values
    across iterations by calculating the mean and also computes CI intervals for median
    and maximum values.

    Inputs:
        df: Datframe containing the values of interest
        group_col: The column name that specifies the AT bin
        value_col: The column name that specifies the metric
                   for which the max and median will be calculated
        bins: List of bins for which this procedure is performed
        size_per_bin: Sample size to which bins are downsampled
        n_boot: Number of subsampling iterations.
        rng: Random number generator.
        ci: Specify which CI should be calculated.

    Returns:
        stats: Dictionary saving averaged median and max values as well
               as the lower and upper precentile specified by ci.
               
    """
    boot_median = np.empty((n_boot, len(bins)))
    boot_max    = np.empty((n_boot, len(bins)))

    split = {b: df.loc[df[group_col] == b, value_col].to_numpy() for b in bins}

    for i in range(n_boot):
        for j, b in enumerate(bins):
            g = split[b]
            idx = rng.integers(0, len(g), size=size_per_bin)  # sample with replacement
            s = g[idx]
            boot_median[i, j] = np.median(s)
            boot_max[i, j]    = np.max(s)
    stats = {}
    stats["x"] = np.array(bins, dtype=float)
    # median
    stats["med_mean"] = boot_median.mean(axis=0)
    stats["med_lo"], stats["med_hi"] = np.percentile(boot_median, ci, axis=0)

    # max
    stats["max_mean"] = boot_max.mean(axis=0)
    stats["max_lo"],  stats["max_hi"]  = np.percentile(boot_max, ci, axis=0)

    return stats

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
############### COMPARING SIMILARITY USING EMD ##################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def bin_filter(RMSD_pooled_by_AT,
               AT_gene_num,
               num_g: int = 200):
    """
    Filter for AT bins by number of genes.
    """
    pooled_by_AT_cp = RMSD_pooled_by_AT.copy()

    for AT, gc in AT_gene_num.items():

        if gc <= num_g:
            pooled_by_AT_cp.pop(AT)

    return pooled_by_AT_cp


def compute_bin_pairwise_emd_resampled(pooled_by_bin: Dict[int, Sequence[float]],
                                       R: int = 100,
                                       replace: bool = True,
                                       random_state: Optional[int] = None
                                       ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute pairwise 1D Earth Mover's Distance (Wasserstein-1) between bins,
    using repeated downsampling to the smallest bin size.

    For each iteration, each bin is subsampled to size of smallest bin, then
    pairwise EMDs are computed. The returned emd_df is theelementwise median 
    across subsampling iterations. 
    This is not performed when comparing distributions of the same bin.

    Input
        pooled_by_bin: Mapping from bin -> RMSD values representing that bin's empirical distribution.
        R: Number of resampling iterations. Default: 100
        replace: Whether to sample with replacement. Default: True
        random_state: Seed for reproducibility. Default: None.

    Returns
        emd_df: Symmetric matrix of pairwise EMD distances between bins,
                where each entry is the median across repeats.
        extras:
            - "n_by_bin": Sample sizes per bin.
            - "min_n": Size of smallest bin.
            - "bins": List of bins in the order used.
            - "replace": Value of replace parameter.
            - "R": number of iterations.
            - "per_repeat": Dictionary that stores all pairwise EMD matrices across subsampling
                            iterations (under key "per_repeat") and bins (under key "bins").
            - "median_by_gap": Stores, for each “bin gap” (k = |bin_i - bin_j|), the median EMD 
                              across all bin pairs that correspond to that gap. Aggregates the 
                              median EMD across subsampling iterations to compute the median per gap.
    """
    # Sort in increasing order
    bins = sorted(pooled_by_bin.keys())

    # Select smallest bin
    n_by_bin = pd.Series({b: len(pooled_by_bin[b]) for b in bins}).sort_index()
    min_n = int(n_by_bin.min())

    rng = np.random.default_rng(random_state)

    num_b = len(bins)
    # Create R layers of num_b x num_b matrices and fill with zeros (basically a stack of matrices)
    # These will be filled with values after each iteration
    per_repeat = np.zeros((R, num_b, num_b), dtype=float)

    # Precompute indices of AT bins to compare (upper triangle)
    pairs = [(i, j) for i in range(num_b) for j in range(num_b)]

    for r in range(R):
        # Resample each bin to min_n
        resampled = {}
        for i, b in enumerate(bins):
            x = pooled_by_bin[b]
            resampled[b] = rng.choice(x, size=min_n, replace=replace)

        # Compute pairwise distances for this repeat
        # Select the respective layer
        mat = per_repeat[r]
        # diagonal stays 0
        for i, j in pairs:
            # pairwise EMD
            bi, bj = bins[i], bins[j]
            d = stats.wasserstein_distance(resampled[bi], resampled[bj])
            # symmetric matrix
            mat[i, j] = mat[j, i] = d

    # Elementwise median across repeats
    # For each positiom (i,j), compute the median across R repeats
    median_mat = np.median(per_repeat, axis=0)
    emd_df = pd.DataFrame(median_mat, index=bins, columns=bins)

    # Median EMD by gap based on the median matrix
    gap_data = []
    for i, j in pairs:
        # Compute the gap
        gap = abs(bins[i] - bins[j])
        # Extends gap_data by the tuple (gap, median value at that gap)
        gap_data.append((gap, emd_df.iat[i, j]))

    df_gap = pd.DataFrame(gap_data, columns=["gap", "emd"])
    # Calculate the median of median emds values per gap -> series with gap as index and median values
    # as values
    median_by_gap = df_gap.groupby("gap")["emd"].median().sort_index()

    extras: Dict[str, Any] = {
        "n_by_bin": n_by_bin,
        "min_n": min_n,
        "bins": bins,
        "replace": replace,
        "R": R,
        "per_repeat": {
            "values": per_repeat,
            "bins": bins,
        },
        "median_by_gap": median_by_gap,
    }

    return emd_df, extras


def _pairs_with_gap_indices(bins: Sequence[int], k: int):
    """
    Helper function for half_gap_compare
    
    Input: 
        bins: List of AT-bins
        k: Bin gap
    
    Returns:
        List of tuples denoting the indices of the AT-bin
        pair that corresponds to gap k.

    """
    pairs = []
    for a in range(len(bins)):
        for b in range(a + 1, len(bins)):
            if abs(bins[b] - bins[a]) == k:
                # Extend pairs with the indices of bins tha
                # correspond to gap k
                pairs.append((a, b))
    return pairs


def half_gap_compare(extras: Dict[str, Any],
                     lower_bins: Sequence[int],
                     upper_bins: Sequence[int],
                     gaps: Tuple[int, ...] = (1, 2, 3),
                     quantiles: Tuple[float, float] = (0.025, 0.975)
                     ) -> pd.DataFrame:
    """
    Compare gap-level EMD medians between two bin subsets ("lower" vs "upper").

    Behavior:
        For each bin gap k and bin set:
            * Gather all EMD values of pairs corresponding to that gap in each repeat
            * Compute the median EMD value at that gap across the pairs within a repeat
            * Calculate the overall median per gap of median EMD values across repeats
            * Calculate 95% CI of these gap-level medians
    """
    per_rep_obj = extras.get("per_repeat", None)
    # Obtain the stack of matrices (num_repeats x num_b x num_b)
    stack = np.asarray(per_rep_obj["values"], dtype=float)
    bins_order = list(per_rep_obj.get("bins", None))

    num_repeats = stack.shape[0]
    # saves for each bin associated index in per repeat matrices 
    # important to make sure i use the right values
    idx_of = {b: i for i, b in enumerate(bins_order)}
    num_b = len(bins_order)
    q_lo, q_hi = quantiles
    rows = []

    lower_bins = list(lower_bins)
    upper_bins = list(upper_bins)

    for k in gaps:
        # Lower set
        pairs_lo_bins = _pairs_with_gap_indices(lower_bins, k)
        # Access bin IDs (number based on AT)
        pairs_lo_bins_id = [(lower_bins[a], lower_bins[b]) for (a, b) in pairs_lo_bins]

        # Upper set
        pairs_up_bins = _pairs_with_gap_indices(upper_bins, k)
        pairs_up_bins_id = [(upper_bins[a], upper_bins[b]) for (a, b) in pairs_up_bins]


        # Access the indices corresponding to the bins in the matrices
        lo_pairs_idx = [(idx_of[a], idx_of[b]) for a, b in pairs_lo_bins_id]
        up_pairs_idx = [(idx_of[a], idx_of[b]) for a, b in pairs_up_bins_id]

        # Number of pairs in each set
        n_lo = len(lo_pairs_idx)
        n_up = len(up_pairs_idx)

        # Obtain all EMD values for gap k in the lower set across repeats
        # shape (num_repeats, n_lo)
        lo_rep_vals = np.array(
            [[stack[r, i, j] for (i, j) in lo_pairs_idx] for r in range(num_repeats)],
            dtype=float
        )
        # Calculate the median EMD for that gap in each repeat (multiple bin
        # comparisons can contribute to a gap, one median per repeat)
        lo_rep = np.nanmedian(lo_rep_vals, axis=1)
        # Calculate median for that gap across repeats -> this is the point estimate
        # for the plot
        lo_med = float(np.nanmedian(lo_rep))
        # Calculate lower and upper quantile of medians across repeats for 95% CI
        lo_ci = (
            float(np.nanquantile(lo_rep, q_lo)),
            float(np.nanquantile(lo_rep, q_hi))
        )

        # Same procedure for upper set
        up_rep_vals = np.array(
            [[stack[r, i, j] for (i, j) in up_pairs_idx] for r in range(num_repeats)],
            dtype=float
        )
        up_rep = np.nanmedian(up_rep_vals, axis=1)  # one median per repeat
        up_med = float(np.nanmedian(up_rep))
        up_ci = (
            float(np.nanquantile(up_rep, q_lo)),
            float(np.nanquantile(up_rep, q_hi)),
        )

        rows.append({
            "gap": int(k),
            "n_pairs_lower": int(n_lo),
            "n_pairs_upper": int(n_up),
            "lower_median": lo_med,
            "lower_ci_low": lo_ci[0],
            "lower_ci_high": lo_ci[1],
            "upper_median": up_med,
            "upper_ci_low": up_ci[0],
            "upper_ci_high": up_ci[1],
            "n_repeats": int(num_repeats),
        })

    return pd.DataFrame(rows)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
################## FIND TWO CLUSTER PARTITION ###################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def find_two_cluster_partition_any_n(fas_data):
    """
    Builds for each gene a graph where transcripts are nodes.
    Transcripts/nodes are connected if they have identical
    FA (FAS score: 1). Uses DFS to search for connected
    connected components.

    Returns:
        bool: True if gene has two connected components
        [A, B]: Members of the two connected compontents

    """
    # Gather isoforms of the gene
    tx = list(fas_data.keys())

    # Build graph of transcripts that are mutually identical (1 both directions)
    # so only isoforms with identical FA are connected by an edge
    adj = {t: set() for t in tx}
    for a, b in combinations(tx, 2):
        if (fas_data[a][b] == 1) and (fas_data[b][a] == 1):
            adj[a].add(b)
            adj[b].add(a)

    # Calculate connected components (isoforms that are maximally similar to each other)
    # using DFS
    comps = []
    seen = set()
    # Iterate over isoforms of gene
    for start in tx:
        # Check if we have encountered this isoform before
        # if so, skip to next one
        if start in seen:
            continue
        # append tx to stack
        stack = [start]
        # build component (set) with tx
        comp = {start}
        seen.add(start)
        # As long as stack is not empty
        while stack:
            # retrieve the last added tx (LIFO behavior)
            u = stack.pop()
            # Search for tx wich have identical FA in the adjacency list
            # -> each tx reachable from the current node 
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
            # stack is empty if all tx connected to starting transcript
            # are found -> connected component
        comps.append(comp)
    # There need to be at least two connected components
    if len(comps) != 2:
        return False, None
    A, B = comps
    # Validate that everything worked
    # within component
    for comp in (A, B):
        comp_list = list(comp)
        for a, b in combinations(comp_list, 2):
            if (fas_data[a][b] != 1) and (fas_data[b][a] != 1):
                return False, None

    # between components
    for a in A:
        for b in B:
            if (fas_data[a][b] != 0) and (fas_data[b][a] != 0):
                return False, None

    return True, [A, B]


def subset_fas_data_to_active_ids(fas_data, active_ids):
    """
    Restrict FAS score matrix to the transcrips which are active
    for the gene.

    Returns:
      sub_fas_data if all active_ids are present, else None
    """
    active_ids = [tx for tx in active_ids if tx in fas_data]
    if len(active_ids) < 2:
        return None

    return {
        a: {b: fas_data[a][b] for b in active_ids}
        for a in active_ids
    }


def find_two_cluster_genes_by_txcount_active_only(genes_by_txcount,
                                                  fas_index,
                                                  fas_scores_fullpath,
                                                  gene_cache_with_active_ids):
    """
    Searches for genes which exhibit the pattern required for being able
    to achieve a RMSD of 1. See also function find_two_cluster_partition_any_n
    for details.
    Considers only the active transcripts of a gene when finding connected components.

    Inputs:
      genes_by_txcount: Dictionary where key is AT bins and values are list of genes
                        within AT bin. Requires that spice.calc_obs_metrics has been run
                        and can be then retrieved from the grouped_by_AT output.
      fas_index: The loaded FAS index dictionary
      fas_scores_fullpath: Path to FAS score matrices
      gene_cache_with_active_ids: output of spice.precompute_ewfd_fixed_AT

    Returns:
      hits_by_bin: Dictionary where key is AT bin and values a list of tuples denoting the
                   hits.
    """
    base = Path(fas_scores_fullpath)
    hits_by_bin = {}

    for tx_count, genes in genes_by_txcount.items():
        bin_hits = []

        for gene in genes:
            # Get file where FAS scores are stored from index
            fas_file = fas_index.get(gene)
            # Get gene information in the current comparison
            cache_entry = gene_cache_with_active_ids.get(gene)
            # Retreive fas_ids of active transcripts
            active_ids = cache_entry.get("fas_ids")

            with open(base / fas_file, "r") as f:
                fas_json = json.load(f)

            fas_data = fas_json[gene]

            # Restrict to active transcripts only
            fas_data_active = subset_fas_data_to_active_ids(fas_data, active_ids)
            if not fas_data_active:
                continue

            ok, sets_ = find_two_cluster_partition_any_n(fas_data_active)
            if ok:
                bin_hits.append((gene, sets_))

        if bin_hits:
            hits_by_bin[tx_count] = bin_hits

    return hits_by_bin


def ref_dist_summary(summary_dict: Dict[int, Dict[str, List[float]]],
                     return_df: bool = True):
    """
    Create a summary table which reflects how many genes, null RMSD
    and unique null RMSD values from a AT-bin null.

    Input:
        AT_bin_nulls: Nested dictionary storing all null values and
                      genes for each gene-mixed AT-bin null.
    Returns: 
        info: Dictioanry with AT as keys AT and values tuple of 
              num_genes (number of genes in this AT bin), num_values
                   (number of RMSD values), num_u_values (number of unique values)
        null_info_df: Dataframe of the info dictionary.
    """

    info_df = {"AT": list(),
               "num_genes": list(),
               "num_values": list(),
               "num_u_values": list()
               }
    info = {key:tuple for key in summary_dict.keys()}

    for at, data in sorted(summary_dict.items()):
        num_unique_genes = len(set(data["genes"]))
        values = len(data["rmsd"])
        unique_values = len(set(data["rmsd"]))
        info[at] = (num_unique_genes,
                         values,
                         unique_values)
        info_df["AT"].append(at)
        info_df["num_genes"].append(num_unique_genes)
        info_df["num_values"].append(values)
        info_df["num_u_values"].append(unique_values)
    
    if return_df:
        return pd.DataFrame(info_df), info
    else:
        return info
