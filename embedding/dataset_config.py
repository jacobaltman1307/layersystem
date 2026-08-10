"""
Shared dataset configuration for embedding scripts.

Centralises the CSV path and category list for each supported dataset
so that individual embedding scripts don't duplicate this logic.

Usage:
    from dataset_config import get_dataset_config, add_dataset_args
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "emails": {
        "csv": "cleanedData.csv",
        "categories": ["Crime", "Entertainment", "Politics", "Science"],
        "text_col": "Content",
        "category_col": "Category",
        "id_col": "ID",
    },
    "agnews": {
        "csv": os.path.join("preprocess", "agnews_cleaned.csv"),
        "categories": ["World", "Sports", "Business", "Sci/Tech"],
        "text_col": "Content",
        "category_col": "Category",
        "id_col": "ID",
    },
    "yelp": {
        "csv": os.path.join("preprocess", "yelp_cleaned.csv"),
        "categories": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        "text_col": "Content",
        "category_col": "Category",
        "id_col": "ID",
    },
}


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add the --dataset CLI argument to an argparse parser."""
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="emails",
        help=f"Dataset to embed. Choices: {list(DATASETS.keys())} (default: emails)",
    )


def get_dataset_config(dataset_name: str) -> dict:
    """Return the config dict for a given dataset name.

    Raises KeyError if the dataset name is not registered.
    """
    if dataset_name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name]
