#!/usr/bin/env python3
"""
export_raw_metrics.py

Extracts and compiles raw evaluation metric data across all grid search runs.
Aggregates results across runs (computing mean and standard deviation) for each
(dataset, embedding_model, method, epsilon, output_dimension) configuration cell,
and exports the result to an analysis-ready tidy CSV.

Usage:
    python export_raw_metrics.py runs/
    python export_raw_metrics.py runs/ -o output/compiled/raw_metrics_tidy.csv
    python export_raw_metrics.py runs/ --export-matrices -o output/compiled/raw_metrics_tidy.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from metrics import compute_scaled_human_utility
except ImportError:
    compute_scaled_human_utility = None

# ---------------------------------------------------------------------------
# Metric Definitions & Metadata
# ---------------------------------------------------------------------------

CORE_METRICS = [
    ("continuity",              "Continuity"),
    ("trustworthiness",         "Trustworthiness"),
    ("cluster_ordering",        "Cluster Ordering"),
    ("pearson",                 "Pearson Correlation"),
    ("spearman",                "Spearman Correlation"),
    ("silhouette",              "Silhouette Score"),
    ("procrustes",              "Procrustes Disparity"),
    ("pairwise_distance_kl",    "Pairwise Distance KL"),
    ("average_metrics",         "Average Metrics"),
    ("wall_clock_time",         "CPU Process Time (s)"),
]

HUMAN_UTILITY_METRICS = [
    ("dbscan_clusters",         "DBSCAN # Clusters"),
    ("spatial_entropy",         "Spatial/Image Entropy"),
    ("overplotting_penalty",    "Overplotting / Crowding Penalty"),
    ("hopkins_statistic",       "Hopkins Statistic"),
    ("absolute_difference",     "Abs Diff Distance Consistency"),
    ("estimated_human_utility", "Estimated Human Utility"),
]

ALL_METRICS_INFO = CORE_METRICS + HUMAN_UTILITY_METRICS
ALL_METRIC_KEYS = [k for k, _ in ALL_METRICS_INFO]
SUPERVISED_ALGORITHMS = {"LDA"}


def is_supervised_algorithm(dim_reduct_name):
    """Check if a dimensionality reduction method contains any supervised algorithm (e.g. LDA)."""
    if not dim_reduct_name:
        return False
    parts = dim_reduct_name.split("_")
    return any(p.upper() in SUPERVISED_ALGORITHMS for p in parts)


def _str_from_npz(val):
    """Safely decode a scalar string stored in an .npz file."""
    if val is None:
        return None
    v = np.asarray(val)
    return str(v.item()) if v.ndim == 0 else str(val)


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
    Expected directory layout: runs/<dataset>/<embedding_model>/<dim_reduct>/*.npz
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

    if not (dataset and embedding_model and dim_reduct) and len(parts) >= 4:
        dim_reduct      = dim_reduct      or parts[-2]
        embedding_model = embedding_model or parts[-3]
        dir_dataset     = dir_dataset     or parts[-4]

    # Resolve dataset to canonical name ('agnews', 'emails', 'yelp')
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
    """Load a single gridsearch .npz result file and extract metrics."""
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
        "file_path":         file_path,
        "file_basename":     file_basename,
        "epsilons":          np.asarray(epsilons),
        "output_dimensions": np.asarray(output_dimensions),
        "embedding_model":   embedding_model,
        "dim_reduct":        dim_reduct,
        "dataset":           dataset,
        "metrics":           {},
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


def crawl_and_aggregate(root_dir, verbose=False, skip_supervised=False):
    """
    Crawls root_dir recursively and aggregates runs by:
    dataset -> embedding_model -> dim_reduct -> list of runs.

    Computes both mean and standard deviation grids across runs for each metric.

    Returns
    -------
    aggregated : dict[dataset][embedding_model][dim_reduct] -> dict
        {
            "epsilons": np.ndarray,
            "output_dimensions": np.ndarray,
            "n_runs": int,
            "metrics": dict[metric_key -> np.ndarray (mean across runs)],
            "metrics_mean": dict[metric_key -> np.ndarray (mean across runs)],
            "metrics_std": dict[metric_key -> np.ndarray (std across runs)]
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

    print(f"Crawled {n_files} run result file(s) from '{root_dir}'.")

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

                matching_runs = []
                for r in runs:
                    if np.array_equal(r["epsilons"], epsilons) and np.array_equal(r["output_dimensions"], output_dimensions):
                        matching_runs.append(r)
                    else:
                        print(f"Warning: Run '{r['file_basename']}' in {ds}/{emb}/{dr} has differing grid shape. Skipping.")

                if not matching_runs:
                    continue

                all_keys = set()
                for r in matching_runs:
                    all_keys.update(r["metrics"].keys())

                mean_metrics = {}
                std_metrics = {}

                for m_key in all_keys:
                    m_arrays = [r["metrics"][m_key] for r in matching_runs if m_key in r["metrics"]]
                    if m_arrays:
                        stacked = np.stack(m_arrays, axis=0)
                        mean_metrics[m_key] = np.nanmean(stacked, axis=0)
                        std_metrics[m_key] = np.nanstd(stacked, axis=0, ddof=1 if len(m_arrays) > 1 else 0)

                aggregated[ds][emb][dr] = {
                    "epsilons":          epsilons,
                    "output_dimensions": output_dimensions,
                    "n_runs":             len(matching_runs),
                    "metrics":           mean_metrics,
                    "metrics_mean":      mean_metrics,
                    "metrics_std":       std_metrics,
                }

    return aggregated


# ---------------------------------------------------------------------------
# CSV Exporters
# ---------------------------------------------------------------------------

def _format_cell(val):
    """Format numeric values cleanly for CSV export."""
    if val is None or np.isnan(val):
        return ""
    if np.isinf(val):
        return "inf" if val > 0 else "-inf"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if abs(val) < 1e-12:
            return "0"
        return f"{val:.6g}"
    return str(val)


def export_raw_metrics_csv(
    aggregated,
    output_path="output/compiled/raw_metrics_tidy.csv",
    include_std=True,
    target_metrics=None,
    dataset_filter=None,
    embedding_filter=None,
    method_filter=None
):
    """
    Exports cell-by-cell raw metric data (averaged across runs) into a tidy CSV.

    Columns:
      Dataset, EmbeddingModel, DimReductionMethod, Epsilon, OutputDimension, CompletedRuns,
      <metric>_mean, <metric>_std (if include_std=True)
      or <metric> (if include_std=False)

    Parameters
    ----------
    aggregated : dict
        Aggregated runs structure from crawl_and_aggregate.
    output_path : str
        Target CSV file path.
    include_std : bool
        Whether to include standard deviation columns alongside means.
    target_metrics : list[str] or None
        Optional subset of metric keys to export.
    dataset_filter : set/list or None
        Optional subset of datasets to include.
    embedding_filter : set/list or None
        Optional subset of embedding models to include.
    method_filter : set/list or None
        Optional subset of dimensionality reduction methods to include.
    """
    metric_keys = [k for k in (target_metrics or ALL_METRIC_KEYS) if k in ALL_METRIC_KEYS]

    # Build CSV header
    header = [
        "Dataset",
        "EmbeddingModel",
        "DimReductionMethod",
        "Epsilon",
        "OutputDimension",
        "CompletedRuns"
    ]

    for k in metric_keys:
        if include_std:
            header.append(f"{k}_mean")
            header.append(f"{k}_std")
        else:
            header.append(k)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    row_count = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ds in sorted(aggregated.keys()):
            if dataset_filter and ds not in dataset_filter:
                continue

            for emb in sorted(aggregated[ds].keys()):
                if embedding_filter and emb not in embedding_filter:
                    continue

                for dr in sorted(aggregated[ds][emb].keys()):
                    if method_filter and dr not in method_filter:
                        continue

                    info = aggregated[ds][emb][dr]
                    epsilons = info["epsilons"]
                    dimensions = info["output_dimensions"]
                    n_runs = info["n_runs"]
                    means = info.get("metrics_mean", info.get("metrics", {}))
                    stds = info.get("metrics_std", {})

                    for i, eps in enumerate(epsilons):
                        for j, dim in enumerate(dimensions):
                            row = [ds, emb, dr, eps, dim, n_runs]
                            for k in metric_keys:
                                m_grid = means.get(k, None)
                                s_grid = stds.get(k, None)

                                mean_val = m_grid[i, j] if m_grid is not None else None
                                std_val = s_grid[i, j] if s_grid is not None else None

                                row.append(_format_cell(mean_val))
                                if include_std:
                                    row.append(_format_cell(std_val))

                            writer.writerow(row)
                            row_count += 1

    print(f"  [Raw Metrics CSV] Saved {row_count} rows to: {output_path}")
    return output_path


def export_matrix_csvs(aggregated, output_dir="output/compiled/raw_matrices", target_metrics=None):
    """
    Exports 2D matrix CSVs per metric for each (dataset, embedding, method),
    where rows are epsilons and columns are output dimensions.
    """
    metric_keys = [k for k in (target_metrics or ALL_METRIC_KEYS) if k in ALL_METRIC_KEYS]
    count = 0

    for ds in sorted(aggregated.keys()):
        for emb in sorted(aggregated[ds].keys()):
            for dr in sorted(aggregated[ds][emb].keys()):
                info = aggregated[ds][emb][dr]
                epsilons = info["epsilons"]
                dimensions = info["output_dimensions"]
                means = info.get("metrics_mean", info.get("metrics", {}))

                target_dir = os.path.join(output_dir, ds, emb, dr)
                os.makedirs(target_dir, exist_ok=True)

                for k in metric_keys:
                    grid = means.get(k, None)
                    if grid is None:
                        continue

                    mat_path = os.path.join(target_dir, f"{k}.csv")
                    with open(mat_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Epsilon \\ Dim"] + [str(d) for d in dimensions])
                        for i, eps in enumerate(epsilons):
                            row = [str(eps)] + [_format_cell(grid[i, j]) for j in range(len(dimensions))]
                            writer.writerow(row)
                    count += 1

    print(f"  [Matrix CSVs] Exported {count} matrix files under: {output_dir}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract cell-by-cell raw metric data averaged across runs for each "
            "method, dataset, and embedding model into an analysis-ready tidy CSV."
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
        "-o", "--output",
        default=os.path.join("output", "compiled", "raw_metrics_tidy.csv"),
        help="Output CSV path for the tidy table (default: output/compiled/raw_metrics_tidy.csv)"
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Exclude standard deviation columns (default: standard deviations are included)"
    )
    parser.add_argument(
        "--export-matrices",
        action="store_true",
        help="Also export individual 2D matrix CSVs per metric to output/compiled/raw_matrices/"
    )
    parser.add_argument(
        "-d", "--dataset",
        nargs="+",
        default=None,
        help="Filter by specific dataset name(s)"
    )
    parser.add_argument(
        "-e", "--embedding",
        nargs="+",
        default=None,
        help="Filter by specific embedding model name(s)"
    )
    parser.add_argument(
        "-m", "--method",
        nargs="+",
        default=None,
        help="Filter by specific dimensionality reduction method(s)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRIC_KEYS,
        default=None,
        help="Export only specific metric keys (default: all 14 metrics)"
    )
    parser.add_argument(
        "-ss", "--skip-supervised",
        action="store_true",
        help="Skip supervised dimensionality reduction methods (e.g. LDA)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose file loading logs"
    )

    args = parser.parse_args()
    crawl_target = args.crawl_dir or args.root_dir

    print(f"Starting raw metrics extraction from: {crawl_target}\n")
    aggregated = crawl_and_aggregate(
        root_dir=crawl_target,
        verbose=args.verbose,
        skip_supervised=args.skip_supervised
    )

    if not aggregated:
        print("No valid grid search runs found to process.")
        sys.exit(0)

    dataset_filter = set(args.dataset) if args.dataset else None
    embedding_filter = set(args.embedding) if args.embedding else None
    method_filter = set(args.method) if args.method else None

    export_raw_metrics_csv(
        aggregated=aggregated,
        output_path=args.output,
        include_std=not args.no_std,
        target_metrics=args.metrics,
        dataset_filter=dataset_filter,
        embedding_filter=embedding_filter,
        method_filter=method_filter
    )

    if args.export_matrices:
        matrix_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", "raw_matrices")
        export_matrix_csvs(aggregated=aggregated, output_dir=matrix_dir, target_metrics=args.metrics)

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
