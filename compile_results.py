import argparse
import os
import sys
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from metrics import compute_scaled_human_utility
except ImportError:
    compute_scaled_human_utility = None

# ---------------------------------------------------------------------------
# Metric Definitions & Metadata
# ---------------------------------------------------------------------------

METRICS_INFO = [
    ("continuity",           "Continuity"),
    ("trustworthiness",      "Trustworthiness"),
    ("cluster_ordering",     "Cluster Ordering"),
    ("pearson",              "Pearson Correlation"),
    ("spearman",             "Spearman Correlation"),
    ("silhouette",           "Silhouette Score"),
    ("procrustes",           "Procrustes Disparity"),
    ("pairwise_distance_kl", "Pairwise Distance KL"),
    ("average_metrics",      "Average Metrics"),
    ("wall_clock_time",      "CPU Process Time (s)"),
]

HUMAN_UTILITY_METRICS_INFO = [
    ("dbscan_clusters",         "DBSCAN # Clusters"),
    ("spatial_entropy",         "Spatial/Image Entropy"),
    ("overplotting_penalty",    "Overplotting / Crowding Penalty"),
    ("hopkins_statistic",       "Hopkins Statistic"),
    ("absolute_difference",     "Abs Diff Distance Consistency"),
    ("estimated_human_utility", "Estimated Human Utility"),
]

ALL_METRICS_INFO = METRICS_INFO + HUMAN_UTILITY_METRICS_INFO

METRIC_RANGES = {
    "continuity":           (0.0,  1.0),
    "trustworthiness":      (0.0,  1.0),
    "cluster_ordering":     (-1.0, 1.0),
    "pearson":              (-1.0, 1.0),
    "spearman":             (-1.0, 1.0),
    "silhouette":           (-1.0, 1.0),
    "procrustes":           (0.0,  1.0),
    "pairwise_distance_kl": (0.0,  None),
    "average_metrics":      (-1.0, 1.0),
    "hopkins_statistic":    (0.0,  1.0),
    "overplotting_penalty": (0.0,  1.0),
    "spatial_entropy":      (0.0,  1.0),
    "estimated_human_utility": (0.0, 1.0),
}

DIVERGING_METRICS = {"pearson", "cluster_ordering", "spearman"}
SUPERVISED_ALGORITHMS = {"LDA"}


def is_supervised_algorithm(dim_reduct_name):
    """Check if a dimensionality reduction method contains any supervised algorithm (e.g. LDA)."""
    if not dim_reduct_name:
        return False
    parts = dim_reduct_name.split("_")
    return any(p.upper() in SUPERVISED_ALGORITHMS for p in parts)


def _cmap_for(metric_key):
    return "PRGn" if metric_key in DIVERGING_METRICS else "viridis"


def _fmt_for(metric_key):
    if metric_key == "dbscan_clusters":
        return ".0f"
    if metric_key in ("procrustes", "pairwise_distance_kl"):
        return ".4f"
    return ".3f"


def _str_from_npz(val):
    """Safely decode a scalar string stored in an .npz file."""
    if val is None:
        return None
    v = np.asarray(val)
    return str(v.item()) if v.ndim == 0 else str(val)


# ---------------------------------------------------------------------------
try:
    from embedding.dataset_config import canonical_dataset_name
except ImportError:
    try:
        from dataset_config import canonical_dataset_name
    except ImportError:
        def canonical_dataset_name(name=None, dir_context=None, categories=None):
            if dir_context:
                d = dir_context.lower()
                if "agnews" in d: return "agnews"
                if "email" in d: return "emails"
                if "yelp" in d: return "yelp"
            if name and str(name).lower() in ("agnews", "news", "ag_news"):
                return "agnews"
            if name and str(name).lower() == "yelp":
                return "yelp"
            return "emails"


def _extract_metadata(file_path, data=None):
    """
    Extract (dataset, embedding_model, dim_reduct) from .npz file or directory structure.
    Always standardizes dataset names to 'agnews', 'emails', or 'yelp'.
    """
    dataset = embedding_model = dim_reduct = None

    if data is not None:
        dataset         = _str_from_npz(data.get("dataset", None))
        embedding_model = _str_from_npz(data.get("embeddingModel", None))
        primary         = _str_from_npz(data.get("primaryDimReductType", None))
        secondary       = _str_from_npz(data.get("secondaryDimReductType", None))

        if primary and secondary and primary != secondary:
            dim_reduct = f"{primary}_{secondary}"
        elif primary:
            dim_reduct = primary

    # Fallback to directory structure: .../runs/<dataset>/<embedding>/<dimreduct>/file.npz
    parts = os.path.normpath(file_path).split(os.sep)
    dir_dataset = None
    try:
        idx = len(parts) - 1 - parts[::-1].index("runs")
        if idx + 3 < len(parts):
            dir_dataset     = parts[idx + 1]
            embedding_model = embedding_model or parts[idx + 2]
            dim_reduct      = dim_reduct      or parts[idx + 3]
    except ValueError:
        pass

    # If still not found and path has at least 3 parent parts
    if not (dataset and embedding_model and dim_reduct) and len(parts) >= 4:
        dim_reduct      = dim_reduct      or parts[-2]
        embedding_model = embedding_model or parts[-3]
        dir_dataset     = dir_dataset     or parts[-4]

    # Resolve dataset to canonical name ('agnews', 'emails', 'yelp')
    # If the file path explicitly sits inside runs/<dataset>/, prefer that directory
    canonical_from_dir = canonical_dataset_name(dir_dataset, dir_context=file_path) if dir_dataset else None
    if canonical_from_dir in ("agnews", "emails", "yelp"):
        dataset = canonical_from_dir
    else:
        dataset = canonical_dataset_name(dataset, dir_context=file_path)

    return (
        dataset         or "UnknownDataset",
        embedding_model or "UnknownEmbedding",
        dim_reduct      or "UnknownDR",
    )



def load_run(file_path):
    """Load a single gridsearch .npz result file."""
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Warning: Failed to load '{file_path}': {e}")
        return None

    output_dimensions = data.get("output_dimensions", data.get("outputDimensions", None))
    if output_dimensions is None:
        for k in data:
            if "dimension" in k.lower():
                output_dimensions = data[k]
                break

    epsilons = data.get("epsilons", None)

    if epsilons is None or output_dimensions is None:
        print(f"Warning: Missing epsilons or output_dimensions in '{file_path}'. Skipping.")
        return None

    dataset, embedding_model, dim_reduct = _extract_metadata(file_path, data)
    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    run_data = {
        "file_path":            file_path,
        "file_basename":        file_basename,
        "epsilons":             np.asarray(epsilons),
        "output_dimensions":    np.asarray(output_dimensions),
        "embedding_model":      embedding_model,
        "dim_reduct":           dim_reduct,
        "dataset":              dataset,
        "metrics":              {},
    }

    for metric_key, _ in ALL_METRICS_INFO:
        metric_data = data.get(metric_key, None)
        if metric_data is None:
            if metric_key == "average_metrics":
                try:
                    metric_data = (
                        data["continuity"] + data["trustworthiness"]
                        + np.abs(data["cluster_ordering"])
                        + np.abs(data["pearson"])
                        + np.abs(data["spearman"])
                        + data["silhouette"]
                    ) / 6.0
                except Exception:
                    metric_data = None
            elif metric_key == "estimated_human_utility":
                if "estimated_human_utility" in data and data["estimated_human_utility"] is not None:
                    metric_data = data["estimated_human_utility"]
                elif compute_scaled_human_utility is not None:
                    hu_dict = {}
                    for hk in ("dbscan_clusters", "spatial_entropy", "overplotting_penalty", "hopkins_statistic", "absolute_difference"):
                        if hk in data and data[hk] is not None:
                            hu_dict[hk] = data[hk]
                        elif hk in run_data["metrics"]:
                            hu_dict[hk] = run_data["metrics"][hk]
                    metric_data = compute_scaled_human_utility(hu_dict)
                else:
                    metric_data = None

        if metric_data is not None:
            run_data["metrics"][metric_key] = np.asarray(metric_data)

    return run_data


# ---------------------------------------------------------------------------
# Crawling & Aggregation
# ---------------------------------------------------------------------------

def crawl_and_aggregate(root_dir, verbose=False, skip_supervised=False):
    """
    Crawls root_dir recursively and aggregates runs by:
    dataset -> embedding_model -> dim_reduct -> list of runs.

    Computes the average metric grid across runs for each (dataset, embedding, method).

    Returns
    -------
    aggregated : dict[dataset][embedding_model][dim_reduct] -> dict
        {
            "epsilons": np.ndarray,
            "output_dimensions": np.ndarray,
            "n_runs": int,
            "metrics": dict[metric_key -> np.ndarray (mean across runs)]
        }
    """
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)

    grouped_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_files = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for fname in sorted(filenames):
            if not fname.endswith(".npz") or fname.startswith("embeddingdata"):
                continue

            file_path = os.path.join(dirpath, fname)
            run = load_run(file_path)
            if run is None:
                continue

            if skip_supervised and is_supervised_algorithm(run["dim_reduct"]):
                if verbose:
                    print(f"  Skipped supervised run: {file_path} -> [{run['dim_reduct']}]")
                continue

            grouped_runs[run["dataset"]][run["embedding_model"]][run["dim_reduct"]].append(run)
            n_files += 1

            if verbose:
                print(f"  Loaded: {file_path} -> [{run['dataset']} | {run['embedding_model']} | {run['dim_reduct']}]")

    print(f"Crawled {n_files} run result file(s) from '{root_dir}'.\n")

    # Aggregate runs per (dataset, embedding_model, dim_reduct)
    aggregated = defaultdict(lambda: defaultdict(dict))

    for ds in sorted(grouped_runs.keys()):
        for emb in sorted(grouped_runs[ds].keys()):
            for dr in sorted(grouped_runs[ds][emb].keys()):
                runs = grouped_runs[ds][emb][dr]
                if not runs:
                    continue

                ref = runs[0]
                epsilons = ref["epsilons"]
                output_dimensions = ref["output_dimensions"]

                # Filter runs that match reference grid dimensions
                matching_runs = []
                for r in runs:
                    if np.array_equal(r["epsilons"], epsilons) and np.array_equal(r["output_dimensions"], output_dimensions):
                        matching_runs.append(r)
                    else:
                        print(f"Warning: Run '{r['file_basename']}' in {ds}/{emb}/{dr} has differing grid shape. Skipping.")

                if not matching_runs:
                    continue

                avg_metrics = {}
                std_metrics = {}
                # Determine all available metric keys
                all_keys = set()
                for r in matching_runs:
                    all_keys.update(r["metrics"].keys())

                for m_key in all_keys:
                    m_arrays = [r["metrics"][m_key] for r in matching_runs if m_key in r["metrics"]]
                    if m_arrays:
                        stacked = np.stack(m_arrays, axis=0)
                        avg_metrics[m_key] = np.nanmean(stacked, axis=0)
                        std_metrics[m_key] = np.nanstd(stacked, axis=0, ddof=1 if len(m_arrays) > 1 else 0)

                aggregated[ds][emb][dr] = {
                    "epsilons": epsilons,
                    "output_dimensions": output_dimensions,
                    "n_runs": len(matching_runs),
                    "metrics": avg_metrics,
                    "metrics_mean": avg_metrics,
                    "metrics_std": std_metrics,
                }

    return aggregated


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def _get_text_color_for_bg(rgb_tuple):
    """Choose white or black text based on background relative luminance."""
    r, g, b = rgb_tuple[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


# ---------------------------------------------------------------------------
# Plot 1: Compiled Method Comparison per (Dataset, Embedding Model)
# ---------------------------------------------------------------------------

def plot_compiled_methods(
    dataset,
    embedding_model,
    methods_data,
    metric_key="average_metrics",
    metric_title="Average Metrics",
    output_path="compiled_algorithms.png"
):
    """
    Compiles each dimensionality reduction method/algorithm for a (dataset, embedding_model)
    into one multi-panel figure.
    """
    methods = sorted(methods_data.keys())
    num_methods = len(methods)
    if num_methods == 0:
        return

    # Determine layout: up to 3 columns (or 4 if >= 8 methods)
    ncols = min(4, num_methods) if num_methods >= 8 else min(3, num_methods)
    nrows = int(np.ceil(num_methods / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 5.2 * nrows),
        squeeze=False
    )

    fig.suptitle(
        f"Compiled Dimensionality Reduction Algorithms — {metric_title}\n"
        f"Dataset: {dataset}  |  Embedding Model: {embedding_model}",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )

    vmin, vmax = METRIC_RANGES.get(metric_key, (None, None))
    cmap = _cmap_for(metric_key)
    fmt = _fmt_for(metric_key)

    for idx, method in enumerate(methods):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        info = methods_data[method]
        grid_vals = info["metrics"].get(metric_key, None)
        n_runs = info["n_runs"]

        if grid_vals is not None:
            sns.heatmap(
                grid_vals,
                xticklabels=info["output_dimensions"],
                yticklabels=info["epsilons"],
                annot=True,
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                ax=ax,
            )
            ax.set_title(f"{method}\n(N={n_runs} run{'s' if n_runs != 1 else ''})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Noise Dimensionality", fontsize=10)
            ax.set_ylabel("Epsilon", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Metric '{metric_key}'\nNot Available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{method} (N={n_runs})", fontsize=12)

    # Hide any unused subplot axes in the grid
    for idx in range(num_methods, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92 if nrows > 1 else 0.88)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Compiled Methods Plot] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Best Method per Grid Square (per Embedding Model)
# ---------------------------------------------------------------------------

def _build_best_grid(methods_data, metric_key="average_metrics"):
    """
    Given methods_data for a single dataset and embedding model:
    Computes for every (epsilon, output_dimension) cell:
      - Best performing method name
      - Best metric score
      - Dict of all method scores at that cell

    Returns
    -------
    eps_list : list of epsilons (sorted)
    dim_list : list of dimensions (sorted descending)
    best_methods : 2D list of str [len(eps)][len(dims)]
    best_scores  : 2D numpy array [len(eps)][len(dims)]
    """
    # Collect all unique epsilons and dimensions
    all_eps = set()
    all_dims = set()

    for method, info in methods_data.items():
        if metric_key in info["metrics"]:
            all_eps.update(info["epsilons"].tolist())
            all_dims.update(info["output_dimensions"].tolist())

    if not all_eps or not all_dims:
        return [], [], [], np.array([])

    # Sort epsilons ascending, dimensions descending (standard grid layout)
    eps_list = sorted(all_eps, key=lambda x: float(x))
    dim_list = sorted(all_dims, key=lambda x: float(x), reverse=True)

    best_methods = [[None for _ in dim_list] for _ in eps_list]
    best_scores = np.full((len(eps_list), len(dim_list)), np.nan)

    for i, eps in enumerate(eps_list):
        for j, dim in enumerate(dim_list):
            candidates = []
            for method, info in methods_data.items():
                if metric_key not in info["metrics"]:
                    continue
                eps_arr = info["epsilons"].tolist()
                dim_arr = info["output_dimensions"].tolist()
                if eps in eps_arr and dim in dim_arr:
                    e_idx = eps_arr.index(eps)
                    d_idx = dim_arr.index(dim)
                    val = info["metrics"][metric_key][e_idx, d_idx]
                    if not np.isnan(val):
                        candidates.append((method, val))

            if candidates:
                # Pick candidate with max score
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_methods[i][j] = candidates[0][0]
                best_scores[i, j] = candidates[0][1]

    return eps_list, dim_list, best_methods, best_scores


def _draw_best_method_ax(ax, eps_list, dim_list, best_methods, best_scores, method_color_map, title=""):
    """Render a single best-method heatmap on a matplotlib axis."""
    n_eps = len(eps_list)
    n_dims = len(dim_list)

    if n_eps == 0 or n_dims == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return

    # Create an RGB image for background colors
    color_grid = np.ones((n_eps, n_dims, 3), dtype=float) * 0.94  # default light gray

    for i in range(n_eps):
        for j in range(n_dims):
            method = best_methods[i][j]
            if method in method_color_map:
                color_grid[i, j] = method_color_map[method][:3]

    ax.imshow(color_grid, aspect="auto", origin="upper")

    # Set tick labels
    ax.set_xticks(np.arange(n_dims))
    ax.set_yticks(np.arange(n_eps))
    ax.set_xticklabels(dim_list, fontsize=9.5 if n_dims > 8 else 10)
    ax.set_yticklabels(eps_list, fontsize=9.5 if n_eps > 8 else 10)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_dims, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_eps, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Adapt font size based on grid density
    if n_dims >= 12 or n_eps >= 12:
        font_size = 7.0
    elif n_dims >= 8 or n_eps >= 8:
        font_size = 8.0
    else:
        font_size = 9.0

    # Add text annotations with method split at '_' and score on next line
    for i in range(n_eps):
        for j in range(n_dims):
            method = best_methods[i][j]
            score = best_scores[i, j]
            if method is not None and not np.isnan(score):
                bg_color = color_grid[i, j]
                txt_color = _get_text_color_for_bg(bg_color)
                # Split compound method names at '_' onto two/separate lines
                method_display = method.replace("_", "\n")
                cell_text = f"{method_display}\n{score:.3f}"
                ax.text(
                    j, i, cell_text,
                    ha="center", va="center",
                    color=txt_color,
                    fontsize=font_size,
                    fontweight="bold",
                    linespacing=0.85
                )
            else:
                ax.text(
                    j, i, "—",
                    ha="center", va="center",
                    color="#888888",
                    fontsize=font_size
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Noise Dimensionality", fontsize=10, labelpad=6)
    ax.set_ylabel("Epsilon", fontsize=10)


def _build_cross_dataset_avg_methods_data(datasets_dict, metric_key="average_metrics"):
    """
    Computes average metrics across multiple datasets for each method.
    """
    combined_methods = defaultdict(dict)
    all_methods = set()
    for ds, m_dict in datasets_dict.items():
        all_methods.update(m_dict.keys())

    for method in all_methods:
        matching_ds_data = [
            m_dict[method] for m_dict in datasets_dict.values()
            if method in m_dict and metric_key in m_dict[method]["metrics"]
        ]
        if not matching_ds_data:
            continue

        ref = matching_ds_data[0]
        epsilons = ref["epsilons"]
        output_dimensions = ref["output_dimensions"]

        valid_arrays = [
            d["metrics"][metric_key] for d in matching_ds_data
            if np.array_equal(d["epsilons"], epsilons) and np.array_equal(d["output_dimensions"], output_dimensions)
        ]
        if valid_arrays:
            combined_methods[method] = {
                "epsilons": epsilons,
                "output_dimensions": output_dimensions,
                "n_runs": sum(d["n_runs"] for d in matching_ds_data),
                "metrics": {metric_key: np.mean(valid_arrays, axis=0)},
            }

    return combined_methods


def build_global_method_color_map(all_methods):
    """
    Build a deterministic, globally consistent mapping from method names to RGB colors.
    Uses a diverse combination of qualitative palettes so every algorithm receives
    the exact same distinct color across all plots throughout the run.
    """
    sorted_methods = sorted(all_methods)

    tab10 = sns.color_palette("tab10")
    set2 = sns.color_palette("Set2")
    tab20b = sns.color_palette("tab20b")
    tab20c = sns.color_palette("tab20c")

    # Canonical default colors for base algorithms
    primary_defaults = {
        "PCA": tab10[0],       # blue
        "TSNE": tab10[3],      # red
        "UMAP": tab10[4],      # purple
        "LDA": tab10[1],       # orange
        "SVD": tab10[2],       # green
        "ISOMAP": tab10[5],    # brown
        "MDS": tab10[6],       # pink
        "SOM": tab10[7],       # gray
        "LLE": tab10[8],       # olive
    }

    full_pool = tab10 + set2 + tab20b + tab20c

    color_map = {}
    used_colors = []

    # 1. Assign primary defaults for base algorithms if present
    for m in sorted_methods:
        if m.upper() in primary_defaults:
            c = primary_defaults[m.upper()]
            color_map[m] = c
            used_colors.append(c)

    # 2. Assign remaining compound/other methods deterministically from the pool
    pool_idx = 0
    for m in sorted_methods:
        if m not in color_map:
            while pool_idx < len(full_pool) and any(np.allclose(full_pool[pool_idx][:3], uc[:3], atol=1e-3) for uc in used_colors):
                pool_idx += 1
            if pool_idx < len(full_pool):
                c = full_pool[pool_idx]
                pool_idx += 1
            else:
                c = full_pool[len(color_map) % len(full_pool)]
            color_map[m] = c
            used_colors.append(c)

    return color_map


def plot_best_methods_for_embedding_model(
    embedding_model,
    datasets_dict,
    all_methods_for_emb,
    method_color_map,
    metric_key="average_metrics",
    metric_title="Average Metrics",
    output_path="best_methods_per_grid.png"
):
    """
    Generates a best-method-for-each-grid-square plot for a given embedding model across its datasets.
    Includes an 'Overall (Dataset Mean)' aggregate subplot if multiple datasets exist.
    Uses the globally consistent method_color_map across all plots.
    """
    datasets = sorted(datasets_dict.keys())
    if not datasets:
        return

    # Build panels list: each dataset, plus aggregate if >= 2 datasets
    panels = [(ds, datasets_dict[ds], f"Dataset: {ds}") for ds in datasets]
    if len(datasets) >= 2:
        cross_ds_data = _build_cross_dataset_avg_methods_data(datasets_dict, metric_key=metric_key)
        if cross_ds_data:
            panels.append(("all_datasets_mean", cross_ds_data, "All Datasets (Mean Average)"))

    # Find max dimensions and epsilons across all panels for dynamic sizing
    max_dims = 0
    max_eps = 0
    for _, p_data, _ in panels:
        for m_info in p_data.values():
            max_dims = max(max_dims, len(m_info.get("output_dimensions", [])))
            max_eps = max(max_eps, len(m_info.get("epsilons", [])))

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    panel_width = max(7.5, 0.85 * max_dims)
    panel_height = max(5.8, 0.65 * max_eps)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_width * ncols, panel_height * nrows + 1.4),
        squeeze=False
    )

    fig.suptitle(
        f"Best Dimensionality Reduction Method per Grid Square — {metric_title}\n"
        f"Embedding Model: {embedding_model}",
        fontsize=15,
        fontweight="bold",
        y=0.99
    )

    # Track win counts per method across all evaluated datasets
    method_win_counts = defaultdict(int)
    total_cells = 0

    for idx, (panel_id, p_methods_data, panel_title) in enumerate(panels):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        eps_list, dim_list, best_methods, best_scores = _build_best_grid(
            p_methods_data, metric_key=metric_key
        )

        # Count wins only from individual datasets (not double-counting aggregate)
        if panel_id != "all_datasets_mean":
            for row in best_methods:
                for m in row:
                    if m is not None:
                        method_win_counts[m] += 1
                        total_cells += 1

        _draw_best_method_ax(
            ax, eps_list, dim_list, best_methods, best_scores,
            method_color_map,
            title=panel_title
        )

    # Hide any unused subplots
    for idx in range(n_panels, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].set_visible(False)

    # Build legend for methods using globally consistent colors
    legend_patches = []
    for m in sorted(all_methods_for_emb):
        wins = method_win_counts[m]
        pct = (wins / total_cells * 100) if total_cells > 0 else 0
        label = f"{m} ({wins} wins, {pct:.1f}%)" if total_cells > 0 else m
        patch = mpatches.Patch(color=method_color_map[m][:3], label=label)
        legend_patches.append(patch)

    if legend_patches:
        n_legend_cols = min(6, len(legend_patches))
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=n_legend_cols,
            frameon=True,
            facecolor="#f8f9fa",
            edgecolor="#cccccc",
            fontsize=10,
            title="Dimensionality Reduction Methods (Win Share across Datasets)",
            title_fontsize=11
        )

    plt.tight_layout()
    # Ensure ample spacing between title, heatmaps, and bottom legend
    bottom_pad = 0.08 + 0.035 * int(np.ceil(len(legend_patches) / min(6, len(legend_patches))))
    fig.subplots_adjust(top=0.91 if nrows > 1 else 0.88, bottom=bottom_pad)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Best Method Plot] Saved: {output_path}")




def export_csv_summaries(aggregated, all_methods_by_emb, output_base, metric_key, metric_title, suffix=""):
    """
    Exports summary CSV reports:
      1. output_base/compiled_summary{suffix}.csv: Tidy overview table.
      2. output_base/best_methods/best_methods_{emb}{suffix}.csv: Best method matrices.
    """
    import csv

    # 1. Tidy summary CSV
    summary_path = os.path.join(output_base, f"compiled_summary{suffix}.csv")
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset",
            "EmbeddingModel",
            "DimReductionMethod",
            "CompletedRuns",
            f"OverallMean_{metric_key}",
            f"MaxScore_{metric_key}",
            "BestEpsilon",
            "BestOutputDimension"
        ])

        for ds in sorted(aggregated.keys()):
            for emb in sorted(aggregated[ds].keys()):
                for dr in sorted(aggregated[ds][emb].keys()):
                    info = aggregated[ds][emb][dr]
                    n_runs = info["n_runs"]
                    grid = info["metrics"].get(metric_key, None)
                    if grid is not None and not np.all(np.isnan(grid)):
                        overall_mean = np.nanmean(grid)
                        max_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
                        best_score = grid[max_idx]
                        best_eps = info["epsilons"][max_idx[0]]
                        best_dim = info["output_dimensions"][max_idx[1]]
                    else:
                        overall_mean = best_score = best_eps = best_dim = "N/A"

                    writer.writerow([
                        ds, emb, dr, n_runs,
                        f"{overall_mean:.4f}" if isinstance(overall_mean, float) else overall_mean,
                        f"{best_score:.4f}" if isinstance(best_score, float) else best_score,
                        best_eps, best_dim
                    ])

    print(f"  [Summary CSV] Saved: {summary_path}")

    # 2. Best methods matrix CSV per embedding model
    for emb in sorted(all_methods_by_emb.keys()):
        emb_datasets_dict = {ds: aggregated[ds][emb] for ds in sorted(aggregated.keys()) if emb in aggregated[ds]}
        best_csv_path = os.path.join(output_base, "best_methods", f"best_methods_{emb}{suffix}.csv")
        os.makedirs(os.path.dirname(best_csv_path) or ".", exist_ok=True)

        with open(best_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for idx, ds in enumerate(sorted(emb_datasets_dict.keys())):
                if idx > 0:
                    writer.writerow([])  # blank separator between datasets

                eps_list, dim_list, best_methods, best_scores = _build_best_grid(
                    emb_datasets_dict[ds], metric_key=metric_key
                )

                writer.writerow([f"{ds} Dataset — Best Method ({metric_title})"] + [f"Dim_{d}" for d in dim_list])
                for i, eps in enumerate(eps_list):
                    row = [f"Eps_{eps}"]
                    for j in range(len(dim_list)):
                        m = best_methods[i][j]
                        s = best_scores[i, j]
                        row.append(f"{m} ({s:.3f})" if m is not None and not np.isnan(s) else "—")
                    writer.writerow(row)

        print(f"  [Best Method CSV] Saved: {best_csv_path}")


def export_best_methods_html(best_methods_dir, image_filenames):
    """
    Generates a simple HTML page containing the best methods images,
    matching the structure in best.html.
    """
    html_path = os.path.join(best_methods_dir, "best.html")
    os.makedirs(best_methods_dir, exist_ok=True)

    img_tags = "\n".join([f"    <img src=\"{fname}\">" for fname in image_filenames])

    html_content = f"""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>
{img_tags}
</body>

</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  [Best Method HTML] Saved: {html_path}")


# ---------------------------------------------------------------------------
# High-Level Orchestrator
# ---------------------------------------------------------------------------

def process_and_compile(root_dir, output_base="output/compiled", human_utility=False, metric=None, skip_supervised=False, verbose=False):
    """
    Crawls root_dir, aggregates runs, and generates:
      1. Compiled method comparison heatmaps per (dataset, embedding_model).
      2. Best method per grid square heatmaps for each embedding_model.
      3. Optionally, the human utility equivalents if human_utility is True.
      4. Summary CSVs for quick analysis and reporting.
      5. Simple best.html page with all generated best methods images.
    """
    if skip_supervised:
        print("Note: Skipping supervised dimensionality reduction algorithms (e.g. LDA).\n")

    aggregated = crawl_and_aggregate(root_dir, verbose=verbose, skip_supervised=skip_supervised)

    if not aggregated:
        print("No valid grid search runs found to compile.")
        return

    # Determine metric configurations to run
    metric_configs = []
    if metric:
        title = dict(ALL_METRICS_INFO).get(metric, metric.replace("_", " ").title())
        metric_configs.append((metric, title, ""))
    else:
        metric_configs.append(("average_metrics", "Average Metrics", ""))

    if human_utility and not metric:
        metric_configs.append(("estimated_human_utility", "Estimated Human Utility", "_human_utility"))

    # Collect global list of all methods and all embedding models
    all_methods_global = set()
    all_methods_by_emb = defaultdict(set)
    for ds, embs in aggregated.items():
        for emb, methods in embs.items():
            all_methods_by_emb[emb].update(methods.keys())
            all_methods_global.update(methods.keys())

    # Build globally consistent color mapping across all methods
    global_method_color_map = build_global_method_color_map(all_methods_global)

    # Track all generated best method images for best.html
    best_method_images = []

    for metric_key, metric_title, suffix in metric_configs:
        print(f"\n{'=' * 70}")
        print(f"Generating Compiled Results for: {metric_title}")
        print(f"{'=' * 70}\n")

        # 1. Generate compiled algorithm plots per (dataset, embedding_model)
        for ds in sorted(aggregated.keys()):
            for emb in sorted(aggregated[ds].keys()):
                methods_data = aggregated[ds][emb]
                out_path = os.path.join(
                    output_base, ds, emb, f"compiled_algorithms{suffix}.png"
                )
                plot_compiled_methods(
                    dataset=ds,
                    embedding_model=emb,
                    methods_data=methods_data,
                    metric_key=metric_key,
                    metric_title=metric_title,
                    output_path=out_path
                )

        # 2. Generate best-method-per-grid-square plot for each embedding model
        for emb in sorted(all_methods_by_emb.keys()):
            emb_datasets_dict = {}
            for ds in sorted(aggregated.keys()):
                if emb in aggregated[ds]:
                    emb_datasets_dict[ds] = aggregated[ds][emb]

            img_name = f"best_methods_{emb}{suffix}.png"
            out_best_path = os.path.join(output_base, "best_methods", img_name)
            best_method_images.append(img_name)

            plot_best_methods_for_embedding_model(
                embedding_model=emb,
                datasets_dict=emb_datasets_dict,
                all_methods_for_emb=all_methods_by_emb[emb],
                method_color_map=global_method_color_map,
                metric_key=metric_key,
                metric_title=metric_title,
                output_path=out_best_path
            )

        # 3. Export CSV summaries
        export_csv_summaries(
            aggregated=aggregated,
            all_methods_by_emb=all_methods_by_emb,
            output_base=output_base,
            metric_key=metric_key,
            metric_title=metric_title,
            suffix=suffix
        )

    # 4. Export simple best.html page with all generated best methods images
    best_methods_dir = os.path.join(output_base, "best_methods")
    export_best_methods_html(best_methods_dir, best_method_images)

    # 5. Export comprehensive cell-by-cell raw metrics CSV
    from export_raw_metrics import export_raw_metrics_csv
    raw_csv_path = os.path.join(output_base, "raw_metrics_tidy.csv")
    export_raw_metrics_csv(
        aggregated=aggregated,
        output_path=raw_csv_path,
        include_std=True,
        target_metrics=[metric] if metric else None
    )

    print(f"\nCompilation complete. All output plots, HTML, and CSVs saved under '{output_base}/'.")



# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile gridsearch results: average runs per dimensionality reduction "
            "method, compile all algorithms into composite plots per dataset/embedding, "
            "and generate best-method-per-grid-square plots for each embedding model."
        )
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="runs",
        help="Root directory of gridsearch runs to crawl (default: runs/)"
    )
    parser.add_argument(
        "-c", "--crawl",
        dest="crawl_dir",
        default=None,
        help="Explicit root directory to crawl recursively (alternative to positional root_dir)"
    )
    parser.add_argument(
        "-d", "--directory",
        dest="single_dir",
        default=None,
        help="Specific directory to load runs from"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.path.join("output", "compiled"),
        help="Base output directory for compiled plots (default: output/compiled/)"
    )
    parser.add_argument(
        "-hu", "--human-utility",
        action="store_true",
        help="Also compile and generate best-method plots for Estimated Human Utility"
    )
    parser.add_argument(
        "-ss", "--skip-supervised", "--no-supervised",
        action="store_true",
        dest="skip_supervised",
        help="Skip compilation of metrics for supervised dimensionality reduction algorithms (e.g. LDA)"
    )
    parser.add_argument(
        "-m", "--metric",
        default=None,
        choices=[k for k, _ in ALL_METRICS_INFO],
        help="Target a specific metric to compile (default: average_metrics)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed loading and metadata extraction logs"
    )
    args = parser.parse_args()

    crawl_target = args.single_dir or args.crawl_dir or args.root_dir

    process_and_compile(
        root_dir=crawl_target,
        output_base=args.output_dir,
        human_utility=args.human_utility,
        metric=args.metric,
        skip_supervised=args.skip_supervised,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
