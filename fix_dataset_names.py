#!/usr/bin/env python3
"""
fix_dataset_names.py

A repeatable, safe, and idempotent migration script to standardize dataset names
across the NoiseLayers project:
  - 'news' -> 'agnews'
  - 'unknown' -> 'emails'
  - 'yelp' -> 'yelp'

This script:
  1. Inspects and updates .npz result files in runs/
  2. Inspects and updates .npz embedding files in /home/jac/dimreduction/
  3. Safely removes stale output directories (output/compiled/news, output/compiled/unknown)
  4. Can be executed repeatedly at any time before or after final experiment runs.
"""

import os
import shutil
import sys
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from embedding.dataset_config import canonical_dataset_name


def update_npz_dataset(file_path: str, new_dataset: str, compressed: bool = False) -> bool:
    """Atomically update the 'dataset' field of an .npz file if different.

    Returns True if the file was modified, False if already correct.
    """
    try:
        loaded = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load '{file_path}': {e}")
        return False

    current_dataset = None
    if "dataset" in loaded:
        val = loaded["dataset"]
        current_dataset = str(np.asarray(val).item()) if np.asarray(val).ndim == 0 else str(val)

    if current_dataset == new_dataset:
        return False

    # Build dictionary of all arrays to preserve everything exactly
    data_dict = {k: loaded[k] for k in loaded.files}
    data_dict["dataset"] = np.array(new_dataset)

    # Save to atomic temp file first
    tmp_path = file_path + ".tmp.npz"
    try:
        if compressed:
            np.savez_compressed(tmp_path, **data_dict)
        else:
            np.savez(tmp_path, **data_dict)
        os.replace(tmp_path, file_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"  [ERROR] Failed saving '{file_path}': {e}")
        return False

    return True


def fix_runs(runs_dir: str):
    """Walk runs/ and standardize dataset in every gridsearch result file."""
    print(f"\nScanning runs directory: {runs_dir}")
    if not os.path.exists(runs_dir):
        print("  Runs directory not found.")
        return 0, 0

    checked = 0
    updated = 0

    for root, dirs, files in os.walk(runs_dir):
        for fname in sorted(files):
            if fname.endswith(".npz") and not fname.endswith(".tmp.npz"):
                file_path = os.path.join(root, fname)
                checked += 1

                # Determine correct dataset based on directory path
                parts = os.path.normpath(file_path).split(os.sep)
                dir_dataset = None
                try:
                    idx = len(parts) - 1 - parts[::-1].index("runs")
                    if idx + 1 < len(parts):
                        dir_dataset = parts[idx + 1]
                except ValueError:
                    pass

                target_dataset = canonical_dataset_name(dir_dataset, dir_context=file_path)

                # Check and update if needed
                if update_npz_dataset(file_path, target_dataset, compressed=False):
                    updated += 1
                    print(f"  [FIXED] {os.path.relpath(file_path, runs_dir)} -> dataset='{target_dataset}'")

    print(f"Runs scan complete: {checked} files checked, {updated} files updated.")
    return checked, updated


def fix_embedding_files(embedding_dir: str):
    """Scan and update embedding .npz files."""
    print(f"\nScanning embedding directory: {embedding_dir}")
    if not os.path.exists(embedding_dir):
        print("  Embedding directory not found.")
        return 0, 0

    checked = 0
    updated = 0

    for fname in sorted(os.listdir(embedding_dir)):
        if fname.startswith("embeddingdata") and fname.endswith(".npz") and not fname.endswith(".tmp.npz"):
            file_path = os.path.join(embedding_dir, fname)
            checked += 1

            try:
                data = np.load(file_path, allow_pickle=True)
                raw_ds = str(data["dataset"]) if "dataset" in data else None
                cats = list(data["categorieslist"]) if "categorieslist" in data else None
            except Exception as e:
                print(f"  [ERROR] Failed to inspect '{fname}': {e}")
                continue

            target_dataset = canonical_dataset_name(raw_ds, dir_context=fname, categories=cats)

            if update_npz_dataset(file_path, target_dataset, compressed=True):
                updated += 1
                print(f"  [FIXED] {fname} -> dataset='{target_dataset}'")

    print(f"Embedding scan complete: {checked} files checked, {updated} files updated.")
    return checked, updated


def cleanup_stale_outputs(output_dir: str):
    """Remove stale 'news' and 'unknown' compiled directories."""
    print(f"\nChecking for stale compiled outputs in: {output_dir}")
    compiled_dir = os.path.join(output_dir, "compiled")
    if not os.path.exists(compiled_dir):
        return

    stale_dirs = ["news", "unknown"]
    for s in stale_dirs:
        target = os.path.join(compiled_dir, s)
        if os.path.exists(target):
            shutil.rmtree(target)
            print(f"  [REMOVED] Stale directory: {target}")


def main():
    print("=" * 60)
    print("NoiseLayers Dataset Standardization & Cleanup")
    print("=" * 60)

    # 1. Fix runs/
    runs_dir = os.path.join(project_root, "runs")
    fix_runs(runs_dir)

    # 2. Fix embedding files in /home/jac/dimreduction
    dimred_dir = os.path.abspath(os.path.join(project_root, ".."))
    fix_embedding_files(dimred_dir)

    # Also check repo root for any local embedding files
    fix_embedding_files(project_root)

    # 3. Cleanup stale compiled directories
    output_dir = os.path.join(project_root, "output")
    cleanup_stale_outputs(output_dir)

    print("\n" + "=" * 60)
    print("Standardization complete. All dataset references are canonical:")
    print("  - agnews")
    print("  - emails")
    print("  - yelp")
    print("=" * 60)


if __name__ == "__main__":
    main()
