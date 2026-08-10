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
    metric_calinski_harabasz,
    metric_overplotting_penalty,
    metric_hopkins_statistic,
)


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

   
    D_low = pdist(loaded_embeddings)
    trustworthiness = metric_trustworthiness(original_embeddings, loaded_embeddings, D_high, D_low)
    continuity = metric_continuity(original_embeddings, loaded_embeddings, D_high, D_low)
    
    cluster_ordering = metric_cluster_ordering(loaded_embeddings, unchanged, loaded_categories)
    pearson = metric_pearson_correlation(loaded_embeddings, unchanged)
    spearman = metric_spearman_correlation(loaded_embeddings, unchanged)
    silhouette = metric_silhouette(loaded_embeddings, loaded_categories)
    absolute = metric_absolute_difference_distance_consistency(loaded_embeddings, unchanged, loaded_categories)
    return continuity, trustworthiness, cluster_ordering, pearson, spearman, silhouette, absolute, loaded_embeddings, unchanged




def gridSearch(embeddingFile, run, dimensionReductionType, secondDimensionReductionType, resolution, embeddingModel, dataset="unknown", plotting=True):
    loaded = np.load(embeddingFile, allow_pickle=True)
    loaded_embeddings = loaded["embeddings"]
    loaded_categories = loaded["categories"]
    loaded_categories_list = loaded["categorieslist"]
    #loaded_texts = loaded["texts"]
    unchanged = loaded_embeddings
    original_embeddings = loaded_embeddings.copy()
    actual_embedding_dim = loaded_embeddings.shape[1]
    
    # Pre-calculate high-dimensional pairwise distance matrix to avoid recalculation inside Factory
    D_high = pdist(original_embeddings)
    

    if resolution == 3:
        epsilons = [1,10,50,100,500,1000,5000,10000]
        outputDimensions = [768,512,256,128,64,32,16,8,4,2]
    elif resolution == 2:
        epsilons = [1,10,50,100,500,1000]
        outputDimensions = [768,384,128,48,8,2]
    elif resolution == 4:
        epsilons = [.1,.5,1,5,10,25,50,100,250,500,1000,2500,5000,10000]
        outputDimensions = [768,512,256,128,96,64,32,16,12,8,6,4,3,2]
    elif resolution == 0:
        epsilons = [1,2]
        outputDimensions = [768,2]    
    else:
        epsilons = [1,10,50,100,500,1000]
        outputDimensions = [768,3,2]

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
    calinski_harabasz = np.zeros((len(epsilons), len(outputDimensions)))
    overplotting_penalty = np.zeros((len(epsilons), len(outputDimensions)))
    hopkins_statistic = np.zeros((len(epsilons), len(outputDimensions)))
    dbscan_clusters = np.zeros((len(epsilons), len(outputDimensions)))
    
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
            
            calinski_val = metric_calinski_harabasz(loaded_emb, dbscan_labels)
            overplot_val = metric_overplotting_penalty(loaded_emb)
            hopkins_val = metric_hopkins_statistic(loaded_emb)

            spatial_entropy[x][y] = entropy_val
            calinski_harabasz[x][y] = calinski_val
            overplotting_penalty[x][y] = overplot_val
            hopkins_statistic[x][y] = hopkins_val


            # Save embeddings for this epsilon/dimension configuration
            save_dict[f"loaded_embeddings_eps_{epsilon}_dim_{outputDim}"] = loaded_emb
            save_dict[f"unchanged_embeddings_eps_{epsilon}_dim_{outputDim}"] = unchanged_emb

            
    print("Continuity:\n", continuity)
    print("Trustworthiness:\n", trustworthiness)
    print("Cluster Ordering:\n", cluster_ordering)
    print("Pearson:\n", pearson)
    print("Spearman:\n", spearman)
    print("Silhouette:\n", silhouette)
    #print("Grid:\n", grid)
    print(len(grid))

    average_metrics = (continuity + trustworthiness + cluster_ordering + pearson + spearman + silhouette) / 6.0
    #average_metrics = (continuity + trustworthiness + np.abs(cluster_ordering) + np.abs(pearson) + np.abs(spearman) + silhouette) / 6.0
    if plotting:
        fig, axes = plt.subplots(2, 4, figsize=(32, 12))

        fig.suptitle(f"Dataset: {dataset}  |  Embedding Model: {embeddingModel}", fontsize=12)
        sns.heatmap(continuity, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[0, 0])
        axes[0, 0].set_title('Continuity')
        axes[0, 0].set_xlabel('Output Dimension')
        axes[0, 0].set_ylabel('Epsilon')

        sns.heatmap(trustworthiness, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=0.0, vmax=1.0, ax=axes[0, 1])
        axes[0, 1].set_title('Trustworthiness')
        axes[0, 1].set_xlabel('Output Dimension')
        axes[0, 1].set_ylabel('Epsilon')

        sns.heatmap(cluster_ordering, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[0, 2])
        axes[0, 2].set_title('Cluster Ordering')
        axes[0, 2].set_xlabel('Output Dimension')
        axes[0, 2].set_ylabel('Epsilon')

        sns.heatmap(pearson, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[0, 3])
        axes[0, 3].set_title('Pearson Correlation')
        axes[0, 3].set_xlabel('Output Dimension')
        axes[0, 3].set_ylabel('Epsilon')

        sns.heatmap(spearman, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[1, 0])
        axes[1, 0].set_title('Spearman Correlation')
        axes[1, 0].set_xlabel('Output Dimension')
        axes[1, 0].set_ylabel('Epsilon')

        sns.heatmap(silhouette, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[1, 1])
        axes[1, 1].set_title('Silhouette Score')
        axes[1, 1].set_xlabel('Output Dimension')
        axes[1, 1].set_ylabel('Epsilon')

        sns.heatmap(average_metrics, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", vmin=-1.0, vmax=1.0, ax=axes[1, 2])
        axes[1, 2].set_title('Average Metrics')
        axes[1, 2].set_xlabel('Output Dimension')
        axes[1, 2].set_ylabel('Epsilon')

        sns.heatmap(wall_clock_time, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".3f", cmap="viridis", ax=axes[1, 3])
        axes[1, 3].set_title('CPU Process Time (s)')
        axes[1, 3].set_xlabel('Output Dimension')
        axes[1, 3].set_ylabel('Epsilon')

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.suptitle(f"DBSCAN Estimated Clusters — {dataset} — {embeddingModel}", fontsize=12)
        sns.heatmap(dbscan_clusters, xticklabels=outputDimensions, yticklabels=epsilons, annot=True, fmt=".0f", cmap="plasma", ax=ax2)
        ax2.set_title('DBSCAN Estimated Clusters')
        ax2.set_xlabel('Output Dimension')
        ax2.set_ylabel('Epsilon')
        fig2.tight_layout()
        fig2.savefig(f"gridsearch_dbscan_clusters_{dataset}_{embeddingModel}_{dimensionReductionType}_{run!s}.png")
        plt.close(fig2)

        plt.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(f"gridsearch_heatmaps_{dataset}_{embeddingModel}_{dimensionReductionType}_{run!s}.png")

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
        "calinski_harabasz": calinski_harabasz,
        "overplotting_penalty": overplotting_penalty,
        "hopkins_statistic": hopkins_statistic,
        "embeddingModel": embeddingModel,
        "primaryDimReductType": dimensionReductionType,
        "secondaryDimReductType": secondDimensionReductionType,
        "dataset": dataset,
    })
    if dimensionReductionType != secondDimensionReductionType:
        dimensionReductionType = f"{dimensionReductionType}_{secondDimensionReductionType}"
    saveResults(save_dict, dimensionReductionType, run, embeddingModel, dataset)
    

def saveResults(save_dict, dimensionReductionType, run, embeddingModel, dataset="unknown"):
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
        help="Search resolution level from 0 to 5 (default: 3)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of results (default: False)"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (e.g., 'news', 'agnews', 'yelp'). Auto-detected from embedding file if not specified."
    )
    args = parser.parse_args()

    plot = False

    if not args.no_plot:
        plot = True

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
            embeddingModel = "UnknownModel"

        # Auto-detect dataset from npz if not specified via CLI
        if args.dataset:
            dataset = args.dataset
        elif "dataset" in loaded:
            ds_val = loaded["dataset"]
            dataset = str(np.asarray(ds_val).item()) if np.asarray(ds_val).ndim == 0 else str(ds_val)
        else:
            dataset = "unknown"
        print(f"Dataset: {dataset}, Embedding Model: {embeddingModel}")
    except FileNotFoundError as e:
        print(f"Error loading embedding model from '{embedding_file}': {e}")
        embeddingModel = "UnknownModel"
        dataset = args.dataset or "unknown"

    # Determine which runs to execute
    if args.run is not None:
        runs_to_execute = [args.run]
    else:
        runs_to_execute = list(range(args.runs))

    #Sets the secondary dimension reduction type if provided
    second_dr_type = args.dr_type
    if args.dr_type_secondary:
        second_dr_type = args.dr_type_secondary

    #Runs primary dimension reduction type
    for run in runs_to_execute:
        gridSearch(embedding_file, run, args.dr_type, second_dr_type, args.resolution, embeddingModel, dataset=dataset, plotting=plot)


if __name__ == "__main__":
    main()