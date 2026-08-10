import argparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from dataset_config import add_dataset_args, get_dataset_config

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Generate Qwen embeddings for a dataset with 8-bit quantization.")
add_dataset_args(parser)
parser.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=True,
    help="Load model with 8-bit quantization using bitsandbytes (default: True)"
)
parser.add_argument(
    "--no-8bit",
    action="store_false",
    dest="load_in_8bit",
    help="Disable 8-bit quantization and load model without quantization"
)
args = parser.parse_args()

cfg = get_dataset_config(args.dataset)
df = pd.read_csv(cfg["csv"])
CategoriesList = cfg["categories"]

#df = df.sample(n=2000, random_state=42).reset_index(drop=True)
IDs = df[cfg["id_col"]]
Sentances = df[cfg["text_col"]]
Categories = df[cfg["category_col"]]

print(f"Device: {device}")
model_id = "Qwen/Qwen3-Embedding-8B"
modelName = model_id.split("/")[-1]

model_kwargs = {}
if args.load_in_8bit and torch.cuda.is_available():
    print("Loading model with 8-bit quantization...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
    }

if model_kwargs:
    model = SentenceTransformer(model_id, model_kwargs=model_kwargs, truncate_dim=768)
else:
    model = SentenceTransformer(model_id, truncate_dim=768).to(device)

embeddings = model.encode(list(Sentances), normalize_embeddings=True, truncate_dim=768)

np.savez_compressed(
    f"embeddingdata{modelName}_{args.dataset}.npz",
    embeddings=embeddings,
    categories=np.array(Categories),
    texts=np.array(Sentances),
    categorieslist=np.array(CategoriesList),
    embeddingModel=modelName,
    dataset=args.dataset
)

print(f"saved data for dataset '{args.dataset}'")

