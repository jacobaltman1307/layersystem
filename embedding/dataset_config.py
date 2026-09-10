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


DATASET_ALIASES = {
    "news": "agnews",
    "ag_news": "agnews",
    "ag-news": "agnews",
    "email": "emails",
    "unknown": "emails",
}


def canonical_dataset_name(name: str = None, dir_context: str = None, categories: list = None) -> str:
    """Normalize a dataset name to one of the canonical names: 'agnews', 'emails', 'yelp'.

    Handles legacy names:
      - 'news' -> 'agnews' (or 'emails' if context/categories indicate emails)
      - 'unknown' -> 'emails'
    """
    # 1. Inspect categories if available
    if categories is not None:
        cat_set = set(categories)
        if any(c in cat_set for c in ["World", "Sports", "Sci/Tech"]):
            return "agnews"
        if any(c in cat_set for c in ["Crime", "Entertainment", "Politics"]):
            return "emails"
        if any(c in cat_set for c in ["1 star", "2 stars", "5 stars"]):
            return "yelp"

    # 2. Inspect dir_context if available
    if dir_context:
        norm_dir = dir_context.lower()
        if "agnews" in norm_dir:
            return "agnews"
        if "email" in norm_dir:
            return "emails"
        if "yelp" in norm_dir:
            return "yelp"

    # 3. Inspect the name itself
    if name is not None:
        cleaned = str(name).strip().lower()
        if cleaned in DATASETS:
            return cleaned
        if cleaned in DATASET_ALIASES:
            # If named 'news' but in an emails context, map to emails
            if cleaned == "news" and dir_context and "email" in dir_context.lower():
                return "emails"
            return DATASET_ALIASES[cleaned]

    # 4. Fallback from dir_context if name was not informative
    if dir_context:
        norm_dir = dir_context.lower()
        if "news" in norm_dir:
            return "agnews"

    return "emails" if name is None or str(name).strip().lower() in ("unknown", "none", "") else str(name).strip().lower()


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

    Resolves aliases ('news' -> 'agnews', 'unknown' -> 'emails').
    Raises KeyError if the dataset name is not recognized.
    """
    canonical = canonical_dataset_name(dataset_name)
    if canonical not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASETS.keys())}"
        )
    return DATASETS[canonical]

