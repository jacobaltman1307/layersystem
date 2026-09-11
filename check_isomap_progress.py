#!/usr/bin/env python3
"""
check_isomap_progress.py
------------------------
Checks the progress of the 36 ISOMAP gridsearch configurations (across runs 0-4 = 180 total runs).
Can output a status table and print the exact SLURM task IDs that are still missing.

Usage:
    python check_isomap_progress.py
    python check_isomap_progress.py --print-missing-tasks
    python check_isomap_progress.py --runs-dir runs/
"""

import argparse
import os
import sys

# The 36 configurations matching run_isomap_gridsearch.slurm
CONFIGS = [
    # --- Vaultgemma-1b ---
    ("embeddingdataVaultgemma-1b_agnews.npz", "Vaultgemma-1b", "agnews", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_agnews.npz", "Vaultgemma-1b", "agnews", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_agnews.npz", "Vaultgemma-1b", "agnews", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_yelp.npz",   "Vaultgemma-1b", "yelp",   "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_yelp.npz",   "Vaultgemma-1b", "yelp",   "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_yelp.npz",   "Vaultgemma-1b", "yelp",   "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_emails.npz", "Vaultgemma-1b", "emails", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_emails.npz", "Vaultgemma-1b", "emails", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdataVaultgemma-1b_emails.npz", "Vaultgemma-1b", "emails", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),

    # --- bert-base-nli-mean-tokens ---
    ("embeddingdatabert-base-nli-mean-tokens_agnews.npz", "bert-base-nli-mean-tokens", "agnews", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens_agnews.npz", "bert-base-nli-mean-tokens", "agnews", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens_agnews.npz", "bert-base-nli-mean-tokens", "agnews", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens_yelp.npz",   "bert-base-nli-mean-tokens", "yelp",   "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens_yelp.npz",   "bert-base-nli-mean-tokens", "yelp",   "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens_yelp.npz",   "bert-base-nli-mean-tokens", "yelp",   "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens.npz",        "bert-base-nli-mean-tokens", "emails", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens.npz",        "bert-base-nli-mean-tokens", "emails", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdatabert-base-nli-mean-tokens.npz",        "bert-base-nli-mean-tokens", "emails", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),

    # --- embeddinggemma-300m ---
    ("embeddingdataembeddinggemma-300m_agnews.npz", "embeddinggemma-300m", "agnews", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m_agnews.npz", "embeddinggemma-300m", "agnews", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m_agnews.npz", "embeddinggemma-300m", "agnews", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m.npz",        "embeddinggemma-300m", "emails", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m.npz",        "embeddinggemma-300m", "emails", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m.npz",        "embeddinggemma-300m", "emails", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m_yelp.npz",   "embeddinggemma-300m", "yelp",   "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m_yelp.npz",   "embeddinggemma-300m", "yelp",   "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdataembeddinggemma-300m_yelp.npz",   "embeddinggemma-300m", "yelp",   "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),

    # --- gemma-3-1b-pt ---
    ("embeddingdatagemma-3-1b-pt_agnews.npz", "gemma-3-1b-pt", "agnews", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_agnews.npz", "gemma-3-1b-pt", "agnews", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_agnews.npz", "gemma-3-1b-pt", "agnews", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_yelp.npz",   "gemma-3-1b-pt", "yelp",   "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_yelp.npz",   "gemma-3-1b-pt", "yelp",   "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_yelp.npz",   "gemma-3-1b-pt", "yelp",   "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_emails.npz", "gemma-3-1b-pt", "emails", "UMAP_ISOMAP", "-t UMAP -t2 ISOMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_emails.npz", "gemma-3-1b-pt", "emails", "ISOMAP_UMAP", "-t ISOMAP -t2 UMAP -s 3 -m 25000"),
    ("embeddingdatagemma-3-1b-pt_emails.npz", "gemma-3-1b-pt", "emails", "ISOMAP",      "-t ISOMAP -s 3 -m 25000"),
]


def check_progress(runs_dir="runs"):
    total_runs = len(CONFIGS) * 5
    completed_count = 0
    missing_tasks = []

    print(f"{'Idx':<4} {'Model':<25} {'Dataset':<8} {'Method':<14} {'Runs 0..4':<12} {'Done':<5}")
    print("-" * 75)

    for cfg_idx, (emb_file, model, dataset, method, args) in enumerate(CONFIGS):
        run_status = []
        cfg_done = 0
        for run_idx in range(5):
            task_id = cfg_idx * 5 + run_idx
            target = os.path.join(runs_dir, dataset, model, method, f"gridsearch_results_{method}_{run_idx}.npz")
            if os.path.exists(target):
                run_status.append("✓")
                cfg_done += 1
                completed_count += 1
            else:
                run_status.append("·")
                missing_tasks.append(task_id)

        status_str = " ".join(run_status)
        print(f"{cfg_idx:<4} {model:<25} {dataset:<8} {method:<14} [{status_str}] {cfg_done}/5")

    print("-" * 75)
    pct = (completed_count / total_runs) * 100 if total_runs else 0
    print(f"Total: {completed_count}/{total_runs} runs completed ({pct:.1f}%). Missing: {len(missing_tasks)} runs.")

    return missing_tasks


def format_slurm_ranges(task_ids):
    """Format a list of integers into SLURM array range syntax (e.g. 0-4,7,10-12)."""
    if not task_ids:
        return ""
    sorted_ids = sorted(task_ids)
    ranges = []
    start = sorted_ids[0]
    prev = sorted_ids[0]

    for x in sorted_ids[1:]:
        if x == prev + 1:
            prev = x
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = x
            prev = x

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def main():
    parser = argparse.ArgumentParser(description="Check progress of ISOMAP gridsearch runs.")
    parser.add_argument("--runs-dir", default="runs", help="Path to runs directory (default: runs)")
    parser.add_argument("--print-missing-tasks", action="store_true", help="Print the SLURM array task specification for missing runs")
    args = parser.parse_args()

    missing = check_progress(args.runs_dir)

    if args.print_missing_tasks and missing:
        slurm_range = format_slurm_ranges(missing)
        print("\nTo submit ONLY missing tasks with sbatch:")
        print(f"sbatch --array={slurm_range}%30 run_isomap_gridsearch.slurm\n")


if __name__ == "__main__":
    main()
