import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

from Layers import NoiseLayer, ReductionLayer
from metrics import (
    metric_absolute_difference_distance_consistency,
    metric_cluster_ordering,
    metric_continuity,
    metric_pearson_correlation,
    metric_silhouette,
    metric_spearman_correlation,
    metric_trustworthiness,
    metric_spatial_entropy,
    metric_overplotting_penalty,
    metric_hopkins_statistic,
    metric_procrustes,
    metric_pairwise_distance_kl,
    compute_scaled_human_utility,
)


try:
    from embedding.dataset_config import canonical_dataset_name
except ImportError:
    try:
        from dataset_config import canonical_dataset_name
    except ImportError:
        def canonical_dataset_name(name=None, dir_context=None, categories=None):
            if name and str(name).lower() in ("agnews", "news", "ag_news"):
                return "agnews"
            if name and str(name).lower() == "yelp":
                return "yelp"
            return "emails"



def Factory(layers,loaded_embeddings,loaded_categories,loaded_categories_list,unchanged,original_embeddings, comparison, D_high):
   
    for i, x in enumerate(layers):
        #print(x["type"])
        if x["type"] == "Noise":
            epsilon = x["parameters"]["epsilon"]
            loaded_embeddings = NoiseLayer(loaded_embeddings,epsilon,False)
        elif x["type"] == "Algorithm":
            algo = x["parameters"]["method"]
            outputDim = x["parameters"]["output_size"]
            loaded_embeddings = ReductionLayer(loaded_embeddings,loaded_categories,algo,outputDim, loaded_categories_list)

    if comparison:
        for _, x in enumerate(layers):
            if x["type"] == "Noise":
                continue
            elif x["type"] == "Algorithm":
                algo = x["parameters"]["method"]
                outputDim = x["parameters"]["output_size"]
                unchanged = ReductionLayer(unchanged,loaded_categories,algo,outputDim, loaded_categories_list)

   
    trustworthiness = metric_trustworthiness(original_embeddings, loaded_embeddings)
    continuity = metric_continuity(original_embeddings, loaded_embeddings)
    
    cluster_ordering = metric_cluster_ordering(loaded_embeddings, unchanged, loaded_categories)
    pearson = metric_pearson_correlation(loaded_embeddings, unchanged)
    spearman = metric_spearman_correlation(loaded_embeddings, unchanged)
    silhouette = metric_silhouette(loaded_embeddings, loaded_categories)
    absolute = metric_absolute_difference_distance_consistency(loaded_embeddings, unchanged, loaded_categories)
    return continuity, trustworthiness, cluster_ordering, pearson, spearman, silhouette, absolute, loaded_embeddings, unchanged




def gridSearch(embeddingFile, run, dimensionReductionType, secondDimensionReductionType, resolution, embeddingModel, dataset="emails", plotting=True, max_samples=None, skip_existing=False):
    dataset = canonical_dataset_name(dataset)
    save_dim_type = dimensionReductionType
    if dimensionReductionType != secondDimensionReductionType:
        save_dim_type = f"{dimensionReductionType}_{secondDimensionReductionType}"
    target_path = f"runs/{dataset}/{embeddingModel}/{save_dim_type}/gridsearch_results_{save_dim_type}_{run!s}.npz"
    if skip_existing and os.path.exists(target_path):
        print(f"Skipping run {run}: {target_path} already exists.")
        return

    loaded = np.load(embeddingFile, allow_pickle=True)
    loaded_embeddings = loaded["embeddings"]
    loaded_categories = loaded["categories"]
    loaded_categories_list = loaded["categorieslist"]

    
    if max_samples is not None and len(loaded_embeddings) > max_samples:
        print(f"Subsampling dataset from {len(loaded_embeddings)} to {max_samples} samples for memory efficiency...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(loaded_embeddings), size=max_samples, replace=False)
        loaded_embeddings = loaded_embeddings[idx]
        loaded_categories = loaded_categories[idx]
        if len(loaded_categories_list) == len(loaded["embeddings"]):
            loaded_categories_list = loaded_categories_list[idx]

    unchanged = loaded_embeddings
    original_embeddings = loaded_embeddings.copy()
    actual_embedding_dim = loaded_embeddings.shape[1]
    
    D_high = None
    

    # Determine number of categories for adaptive resolutions
    if loaded_categories_list is not None and hasattr(loaded_categories_list, '__len__') and 0 < len(loaded_categories_list) < len(loaded_embeddings):
        num_categories = len(loaded_categories_list)
    else:
        num_categories = len(np.unique(loaded_categories))

    max_cat_dim = num_categories - 1
    adaptive_dims = list(range(max_cat_dim, 1, -1)) if max_cat_dim >= 2 else [2]

    if resolution == 3:
        epsilons = [1,10,50,100,500,1000,5000,10000,1000000000]
        outputDimensions = [768,512,256,128,64,32,16,8,4,2]
    elif resolution == 2:
        epsilons = [1,10,50,100,500,1000]
        outputDimensions = [768,384,128,48,8,2]
    elif resolution == 4:
        epsilons = [.1,.5,1,5,10,25,50,100,250,500,1000,2500,5000,10000,1000000000]
        outputDimensions = [768,512,256,128,96,64,32,16,12,8,6,4,3,2]
    elif resolution == 0:
        epsilons = [1,2]
        outputDimensions = [768,2]    
    elif resolution == 5:
        epsilons = [1,10,50,100,500,1000,5000,10000,1000000000]
        outputDimensions = adaptive_dims
    else:
        epsilons = [1,10,50,100,500,1000,1000000000]
        outputDimensions = [768,3,2]

    # Automatically adapt output dimensions for constrained algorithms like LDA and TSNE
    constrained_algos = {"LDA", "TSNE"}
    is_constrained = (
        (isinstance(dimensionReductionType, str) and dimensionReductionType.upper() in constrained_algos) or
        (isinstance(secondDimensionReductionType, str) and secondDimensionReductionType.upper() in constrained_algos)
    )
    if is_constrained or resolution == 5:
        outputDimensions = adaptive_dims
        print(f"Using adaptive resolution for {dimensionReductionType} ({num_categories} categories -> dims {outputDimensions})")

    # Cap output dimensions to the actual embedding dimension
    outputDimensions = [d for d in outputDimensions if d <= actual_embedding_dim]


    grid = np.zeros((len(epsilons), len(outputDimensions)))

    continuity = np.zeros((len(epsilons), len(outputDimensions)))
    trustworthiness = np.zeros((len(epsilons), len(outputDimensions)))
    cluster_ordering = np.zeros((len(epsilons), len(outputDimensions)))
    pearson = np.zeros((len(epsilons), len(outputDimensions)))
    spearman = np.zeros((len(epsilons), len(outputDimensions)))
    silhouette = np.zeros((len(epsilons), len(outputDimensions)))
    wall_clock_time = np.zeros((len(epsilons), len(outputDimensions)))
    absolute_difference = np.zeros((len(epsilons), len(outputDimensions)))
    spatial_entropy = np.zeros((len(epsilons), len(outputDimensions)))
    overplotting_penalty = np.zeros((len(epsilons), len(outputDimensions)))
    hopkins_statistic = np.zeros((len(epsilons), len(outputDimensions)))
    dbscan_clusters = np.zeros((len(epsilons), len(outputDimensions)))
    procrustes = np.zeros((len(epsilons), len(outputDimensions)))
    pairwise_distance_kl = np.zeros((len(epsilons), len(outputDimensions)))
    
    save_dict = {}


    

    for x, epsilon in enumerate(epsilons):
        for y, outputDim in enumerate(outputDimensions):
            print(f"Epsilon: {epsilon}, Output Dim: {outputDim}")
            layers = [
                {
                    "type": "Algorithm",
                    "parameters": {
                        "method": dimensionReductionType,
                        "output_size": outputDim
                    }
                },
                {
                    "type": "Noise",
                    "parameters": {
                        "epsilon": epsilon
                    }
                },
                {
                    "type": "Algorithm",
                    "parameters": {
                        "method": secondDimensionReductionType,
                        "output_size": 2
                    }
                }
            ]
            if outputDim == 768 and actual_embedding_dim == 768:
                layers = layers[1:]

                #def Factory(layers,loaded_embeddings,loaded_categories,loaded_categories_list,unchanged,original_embeddings, comparison):
                # 
            startTime = time.process_time()
            continuityMetric, trustworthinessMetric, cluster_orderingMetric, pearsonMetric, spearmanMetric, silhouetteMetric, absoluteMetric, loaded_emb, unchanged_emb = Factory(layers, loaded_embeddings, loaded_categories, loaded_categories_list, unchanged, original_embeddings, True, D_high)
            endTime = time.process_time()
            grid[x][y] = 1
            continuity[x][y] = continuityMetric
            trustworthiness[x][y] = trustworthinessMetric
            cluster_ordering[x][y] = cluster_orderingMetric
            pearson[x][y] = pearsonMetric[0]
            spearman[x][y] = spearmanMetric[0]
            silhouette[x][y] = silhouetteMetric
            absolute_difference[x][y] = absoluteMetric
            wall_clock_time[x][y] = endTime - startTime

            #This estimates the human visual utility
            dbscan_labels = DBSCAN().fit_predict(loaded_emb)
            n_clusters = len(set(dbscan_labels) - {-1})
            dbscan_clusters[x][y] = n_clusters
            entropy_val = metric_spatial_entropy(loaded_emb)
            overplot_val = metric_overplotting_penalty(loaded_emb)
            hopkins_val = metric_hopkins_statistic(loaded_emb)

            spatial_entropy[x][y] = entropy_val
            overplotting_penalty[x][y] = overplot_val
            hopkins_statistic[x][y] = hopkins_val

            # Evaluation comparing with noise vs without noise
            procrustes[x][y] = metric_procrustes(loaded_emb, unchanged_emb)
            pairwise_distance_kl[x][y] = metric_pairwise_distance_kl(loaded_emb, unchanged_emb)

            # Save embeddings for this epsilon/dimension configuration
            save_dict[f"loaded_embeddings_eps_{epsilon}_dim_{outputDim}"] = loaded_emb
            save_dict[f"unchanged_embeddings_eps_{epsilon}_dim_{outputDim}"] = unchanged_emb

            
    print("Continuity:\n", continuity)
    print("Trustworthiness:\n", trustworthiness)
    print("Cluster Ordering:\n", cluster_ordering)
    print("Pearson:\n", pearson)
    print("Spearman:\n", spearman)
    print("Silhouette:\n", silhouette)
    print("Procrustes Disparity:\n", procrustes)
    print("Pairwise Distance KL:\n", pairwise_distance_kl)
    #print("Grid:\n", grid)
    print(len(grid))

    average_metrics = (continuity + trustworthiness + cluster_ordering + pearson + spearman + silhouette) / 6.0
    #average_metrics = (continuity + trustworthiness + np.abs(cluster_ordering) + np.abs(pearson) + np.abs(spearman) + silhouette) / 6.0

    estimated_human_utility = compute_scaled_human_utility({
        "dbscan_clusters": dbscan_clusters,
        "spatial_entropy": spatial_entropy,
        "overplotting_penalty": overplotting_penalty,
        "hopkins_statistic": hopkins_statistic,
        "absolute_difference": absolute_difference,
    })
    if estimated_human_utility is not None:
        print("Estimated Human Utility (Scaled):\n", estimated_human_utility)

    if plotting:
        fig, axes = plt.subplots(2, 5, figsize=(40, 12))

        fig.suptitle(f"Dataset: {dataset}  |  Embedding Model: {embeddingModel}", fontsize=12)
        sns.heatmap(continuity, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[0, 0])
        axes[0, 0].set_title('Continuity')
        axes[0, 0].set_xlabel('Noise Dimensionality')
        axes[0, 0].set_ylabel('Epsilon')

        sns.heatmap(trustworthiness, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[0, 1])
        axes[0, 1].set_title('Trustworthiness')
        axes[0, 1].set_xlabel('Noise Dimensionality')
        axes[0, 1].set_ylabel('Epsilon')

        sns.heatmap(cluster_ordering, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[0, 2])
        axes[0, 2].set_title('Cluster Ordering')
        axes[0, 2].set_xlabel('Noise Dimensionality')
        axes[0, 2].set_ylabel('Epsilon')

        sns.heatmap(silhouette, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[0, 3])
        axes[0, 3].set_title('Silhouette')
        axes[0, 3].set_xlabel('Noise Dimensionality')
        axes[0, 3].set_ylabel('Epsilon')

        sns.heatmap(procrustes, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".4f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[0, 4])
        axes[0, 4].set_title('Procrustes Disparity')
        axes[0, 4].set_xlabel('Noise Dimensionality')
        axes[0, 4].set_ylabel('Epsilon')

        sns.heatmap(pearson, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[1, 0])
        axes[1, 0].set_title('Pearson')
        axes[1, 0].set_xlabel('Noise Dimensionality')
        axes[1, 0].set_ylabel('Epsilon')

        sns.heatmap(spearman, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[1, 1])
        axes[1, 1].set_title('Spearman')
        axes[1, 1].set_xlabel('Noise Dimensionality')
        axes[1, 1].set_ylabel('Epsilon')

        sns.heatmap(pairwise_distance_kl, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".4f", cmap="viridis", vmin=0.0, ax=axes[1, 2])
        axes[1, 2].set_title('Pairwise Distance KL')
        axes[1, 2].set_xlabel('Noise Dimensionality')
        axes[1, 2].set_ylabel('Epsilon')

        sns.heatmap(wall_clock_time, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".2f", cmap="viridis", ax=axes[1, 3])
        axes[1, 3].set_title('Time (s)')
        axes[1, 3].set_xlabel('Noise Dimensionality')
        axes[1, 3].set_ylabel('Epsilon')

        sns.heatmap(average_metrics, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[1, 4])
        axes[1, 4].set_title('Average Metrics')
        axes[1, 4].set_xlabel('Noise Dimensionality')
        axes[1, 4].set_ylabel('Epsilon')

        plt.tight_layout()
        plt.show()

    # Save metrics and embeddings to npz file
    save_dict.update({
        "epsilons": np.array(epsilons),
        "output_dimensions": np.array(outputDimensions),
        "continuity": continuity,
        "trustworthiness": trustworthiness,
        "cluster_ordering": cluster_ordering,
        "pearson": pearson,
        "spearman": spearman,
        "silhouette": silhouette,
        "absolute_difference": absolute_difference,
        "wall_clock_time": wall_clock_time,
        "average_metrics": average_metrics,
        "dbscan_clusters": dbscan_clusters,
        "spatial_entropy": spatial_entropy,
        "overplotting_penalty": overplotting_penalty,
        "hopkins_statistic": hopkins_statistic,
        "estimated_human_utility": estimated_human_utility,
        "procrustes": procrustes,
        "pairwise_distance_kl": pairwise_distance_kl,
        "embeddingModel": embeddingModel,
        "primaryDimReductType": dimensionReductionType,
        "secondaryDimReductType": secondDimensionReductionType,
        "dataset": dataset,
    })
    if dimensionReductionType != secondDimensionReductionType:
        dimensionReductionType = f"{dimensionReductionType}_{secondDimensionReductionType}"
    saveResults(save_dict, dimensionReductionType, run, embeddingModel, dataset)
    

def saveResults(save_dict, dimensionReductionType, run, embeddingModel, dataset="emails"):
    dataset = canonical_dataset_name(dataset)
    path = f"runs/{dataset}/{embeddingModel}/{dimensionReductionType}/"
    if not os.path.exists(path):
         os.makedirs(path)
    np.savez(f"{path}gridsearch_results_{dimensionReductionType}_{run!s}.npz", **save_dict)
    print(f"Saved results and embeddings to {path}gridsearch_results_{dimensionReductionType}_{run!s}.npz")


def main():
    parser = argparse.ArgumentParser(description="Perform a grid search comparing dimensionality reduction and noise levels.")
    parser.add_argument(
        "-f", "--embedding-file",
        default="embeddingdataBERT.npz",
        help="Path to the embedding .npz file (default: embeddingdataBERT.npz)"
    )
    parser.add_argument(
        "-r", "--runs",
        type=int,
        default=5,
        help="Number of runs to execute (default: 5)"
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Specific run ID to execute (0-indexed). If specified, runs only this run instead of looping through all runs."
    )
    parser.add_argument(
        "-t", "--dr-type",
        default="PCA",
        help="Primary dimension reduction type (default: PCA)"
    )
    parser.add_argument(
        "-t2", "--dr-type-secondary",
        default=None,
        help="Secondary optional dimension reduction type (default: None)"
    )
    parser.add_argument(
        "-s", "--resolution",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=3,
        help="Search resolution level from 0 to 5 (default: 3, 5=adaptive for LDA/TSNE)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of results (default: False)"
    )
    parser.add_argument(
        "--dataset",
        choices=["agnews", "emails", "yelp"],
        default=None,
        help="Dataset name ('agnews', 'emails', 'yelp'). Auto-detected from embedding file if not specified."
    )
    parser.add_argument(
        "-m", "--max-samples",
        type=int,
        default=None,
        help="Subsample dataset to N items max for memory-heavy algorithms like Isomap/MDS (default: None)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip execution if the target output .npz file already exists"
    )
    args = parser.parse_args()

    plot = False

    if not args.no_plot:
        plot = True

    # Sets the secondary dimension reduction type if provided
    second_dr_type = args.dr_type_secondary if args.dr_type_secondary else args.dr_type
    save_dim_type = f"{args.dr_type}_{second_dr_type}" if args.dr_type != second_dr_type else args.dr_type

    # Determine which runs to execute
    if args.run is not None:
        runs_to_execute = [args.run]
    else:
        runs_to_execute = list(range(args.runs))

    # Early check if all requested runs already exist
    fname = os.path.basename(args.embedding_file).lower()
    inferred_ds = args.dataset or ("agnews" if "agnews" in fname or "news" in fname else ("yelp" if "yelp" in fname else "emails"))
    inferred_ds = canonical_dataset_name(inferred_ds)

    if "vaultgemma" in fname:
        inferred_model = "Vaultgemma-1b"
    elif "bert" in fname:
        inferred_model = "bert-base-nli-mean-tokens"
    elif "embeddinggemma" in fname or "300m" in fname:
        inferred_model = "embeddinggemma-300m"
    elif "gemma-3" in fname or "gemma3" in fname:
        inferred_model = "gemma-3-1b-pt"
    elif "qwen" in fname:
        inferred_model = "Qwen3-Embedding-8B"
    else:
        inferred_model = None

    if args.skip_existing and inferred_model:
        remaining_runs = [
            r for r in runs_to_execute
            if not os.path.exists(f"runs/{inferred_ds}/{inferred_model}/{save_dim_type}/gridsearch_results_{save_dim_type}_{r}.npz")
        ]
        if not remaining_runs:
            print(f"Skipping: all requested runs ({runs_to_execute}) for {inferred_ds}/{inferred_model}/{save_dim_type} already exist.")
            return
        runs_to_execute = remaining_runs

    # Load the embedding model name and dataset from the npz file
    embedding_file = args.embedding_file
    if not os.path.exists(embedding_file):
        print(f"Error: Embedding file '{embedding_file}' does not exist.")
        return

    try:
        loaded = np.load(embedding_file, allow_pickle=True)
        if "embeddingModel" in loaded:
            emb_model_val = loaded["embeddingModel"]
            if isinstance(emb_model_val, np.ndarray):
                embeddingModel = str(emb_model_val.item()) if emb_model_val.ndim == 0 else str(emb_model_val[0])
            else:
                embeddingModel = str(emb_model_val)
        else:
            embeddingModel = inferred_model or "UnknownModel"

        # Auto-detect dataset from npz or filename if not specified via CLI
        cats_list = list(loaded["categorieslist"]) if "categorieslist" in loaded else None
        if args.dataset:
            dataset = canonical_dataset_name(args.dataset, dir_context=embedding_file, categories=cats_list)
        elif "dataset" in loaded:
            ds_val = loaded["dataset"]
            raw_ds = str(np.asarray(ds_val).item()) if np.asarray(ds_val).ndim == 0 else str(ds_val)
            dataset = canonical_dataset_name(raw_ds, dir_context=embedding_file, categories=cats_list)
        else:
            dataset = canonical_dataset_name(inferred_ds, dir_context=embedding_file, categories=cats_list)

        print(f"Dataset: {dataset}, Embedding Model: {embeddingModel}")
    except FileNotFoundError as e:
        print(f"Error loading embedding model from '{embedding_file}': {e}")
        embeddingModel = inferred_model or "UnknownModel"
        dataset = canonical_dataset_name(args.dataset or inferred_ds or "emails")

    #Runs primary dimension reduction type
    for run in runs_to_execute:
        gridSearch(embedding_file, run, args.dr_type, second_dr_type, args.resolution, embeddingModel, dataset=dataset, plotting=plot, max_samples=args.max_samples, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()