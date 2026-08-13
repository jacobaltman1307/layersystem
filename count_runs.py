import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

# Written in part by Claude Opus 4.6
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_from_npz(val):
    """Safely decode a scalar string stored in an .npz file."""
    if val is None:
        return None
    v = np.asarray(val)
    return str(v.item()) if v.ndim == 0 else str(val)


def _extract_metadata(file_path):
    """
    Try to extract (dataset, embedding_model, primary_dim_reduct) from
    a .npz file.  Falls back to parsing the directory structure if the
    metadata keys are missing inside the file.

    Expected directory layout (from gridsearch.py saveResults):
        runs/<dataset>/<embeddingModel>/<dimReductionType>/*.npz
    """
    dataset = embedding_model = dim_reduct = None

    # --- Try reading metadata stored inside the file ---
    try:
        data = np.load(file_path, allow_pickle=True)
        dataset         = _str_from_npz(data.get("dataset",                None))
        embedding_model = _str_from_npz(data.get("embeddingModel",         None))
        primary         = _str_from_npz(data.get("primaryDimReductType",   None))
        secondary       = _str_from_npz(data.get("secondaryDimReductType", None))

        # Build the combined dim-reduct label the same way gridsearch.py
        # names the directory (line 280-281).
        if primary and secondary and primary != secondary:
            dim_reduct = f"{primary}_{secondary}"
        elif primary:
            dim_reduct = primary
    except Exception:
        pass  # fall through to directory-based detection

    # --- Fallback: derive from directory path ---
    # Expected: .../runs/<dataset>/<embedding>/<dimreduct>/file.npz
    parts = os.path.normpath(file_path).split(os.sep)
    # Find "runs" in the path and take the three parts after it
    try:
        idx = len(parts) - 1 - parts[::-1].index("runs")  # last occurrence
        if idx + 3 < len(parts):
            dataset         = dataset         or parts[idx + 1]
            embedding_model = embedding_model or parts[idx + 2]
            dim_reduct      = dim_reduct      or parts[idx + 3]
    except ValueError:
        pass  # "runs" not in path

    return (
        dataset         or "Unknown",
        embedding_model or "Unknown",
        dim_reduct      or "Unknown",
    )


# ---------------------------------------------------------------------------
# Crawl & count
# ---------------------------------------------------------------------------

def crawl_and_count(root_dir, verbose=False):
    """
    Walk *root_dir* recursively.  For every .npz file that looks like a
    gridsearch result (i.e. not an embeddingdata* file), extract metadata
    and tally completed runs.

    Returns
    -------
    counts : dict[dataset][dim_reduct][embedding_model] -> int
    """
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)

    # counts[dataset][dim_reduct][embedding_model] = count
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    n_files = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for fname in sorted(filenames):
            if not fname.endswith(".npz"):
                continue
            if fname.startswith("embeddingdata"):
                continue

            file_path = os.path.join(dirpath, fname)
            dataset, embedding, dim_reduct = _extract_metadata(file_path)
            counts[dataset][dim_reduct][embedding] += 1
            n_files += 1

            if verbose:
                print(f"  {file_path}")
                print(f"    -> dataset={dataset}  embedding={embedding}  dim_reduct={dim_reduct}")

    print(f"Scanned {n_files} result file(s) under '{root_dir}'.\n")
    return counts


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(counts, output_path):
    """
    Write a single CSV file with one table per dataset.

    Layout (matches the user's reference image):

        <Dataset> Dataset
        ,Bert,EmbGemma,Gemma3,Vault,...
        PCA,5,5,5,0,...
        SVD,5,5,5,0,...
        ...
        <blank line>
        <Next Dataset> Dataset
        ...
    """
    # Collect the global superset of embeddings across all datasets so
    # columns are consistent, but order each dataset's embeddings in
    # discovery order as a fallback.
    all_embeddings = []
    seen = set()
    for dataset in sorted(counts.keys()):
        for dim_reduct in sorted(counts[dataset].keys()):
            for emb in counts[dataset][dim_reduct]:
                if emb not in seen:
                    all_embeddings.append(emb)
                    seen.add(emb)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for i, dataset in enumerate(sorted(counts.keys())):
            if i > 0:
                writer.writerow([])  # blank separator between datasets

            # Header row for this dataset
            writer.writerow([f"{dataset} Dataset"] + all_embeddings)

            dim_reducts = sorted(counts[dataset].keys())
            for dr in dim_reducts:
                row = [dr]
                for emb in all_embeddings:
                    row.append(counts[dataset][dr].get(emb, 0))
                writer.writerow(row)

    print(f"Saved run counts to '{output_path}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crawl a directory of gridsearch runs and count completed "
            ".npz result files per dataset, embedding model, and "
            "dimension-reduction method.  Outputs a single CSV file "
            "with one table per dataset."
        )
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="runs",
        help="Root directory to crawl (default: runs/)"
    )
    parser.add_argument(
        "-o", "--output",
        default="run_counts.csv",
        help="Output CSV file path (default: run_counts.csv)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each file and its detected metadata"
    )
    args = parser.parse_args()

    counts = crawl_and_count(args.root_dir, verbose=args.verbose)

    if not counts:
        print("No completed runs found.")
        sys.exit(0)

    write_csv(counts, args.output)

    # Also print a summary to the console
    all_embeddings = []
    seen = set()
    for dataset in sorted(counts.keys()):
        for dr in sorted(counts[dataset].keys()):
            for emb in counts[dataset][dr]:
                if emb not in seen:
                    all_embeddings.append(emb)
                    seen.add(emb)

    for dataset in sorted(counts.keys()):
        print(f"\n{'=' * 60}")
        print(f"  {dataset} Dataset")
        print(f"{'=' * 60}")

        # Column header
        col_width = max(len(e) for e in all_embeddings) + 2
        header = f"{'':15s}" + "".join(f"{e:>{col_width}s}" for e in all_embeddings)
        print(header)
        print("-" * len(header))

        for dr in sorted(counts[dataset].keys()):
            row = f"{dr:15s}"
            for emb in all_embeddings:
                val = counts[dataset][dr].get(emb, 0)
                row += f"{val:>{col_width}d}"
            print(row)

    total = sum(
        c
        for ds in counts.values()
        for dr in ds.values()
        for c in dr.values()
    )
    print(f"\nTotal completed runs: {total}")


if __name__ == "__main__":
    main()
