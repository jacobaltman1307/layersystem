import json
import sys
#import time
import numpy as np
from Layers import NoiseLayer, ReductionLayer
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import metric_pearson_correlation, metric_cluster_ordering, metric_continuity
from scipy.spatial.distance import pdist
#START = 0 
#END = 0 

def Factory(layers, embeddingFile, proven, comparison, metrics):
    #START = time.time()
    loaded = np.load(embeddingFile, allow_pickle=True)
    loaded_embeddings = loaded["embeddings"]
    loaded_categories = loaded["categories"]
    loaded_categories_list = loaded["categorieslist"]
    #loaded_texts = loaded["texts"]
    dataset = str(np.asarray(loaded["dataset"]).item()) if "dataset" in loaded else "unknown"
    unchanged = loaded_embeddings
    original_embeddings = loaded_embeddings.copy()
    
    metric = []

    for i, x in enumerate(layers):
        #print(x["type"])
        if x["type"] == "Noise":
            epsilon = x["parameters"]["epsilon"]
            loaded_embeddings = NoiseLayer(loaded_embeddings,epsilon,proven)
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

    if metrics:
        continunity = metric_continuity(original_embeddings, loaded_embeddings)
        #print(f"Continuity: {continunity}")
        
        cluster_ordering = metric_cluster_ordering(loaded_embeddings, unchanged, loaded_categories)
        pearson = metric_pearson_correlation(loaded_embeddings, unchanged)

        metric.append(continunity)
        metric.append(cluster_ordering)
        metric.append(pearson)


    

    #print(layers)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Dataset: {dataset}", fontsize=14)

    sns.scatterplot(x=unchanged[:, 0], y=unchanged[:, 1], hue=loaded_categories, alpha=0.6, ax=axes[0])
    axes[0].set_title('Without Noise')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')


    sns.scatterplot(x=loaded_embeddings[:, 0], y=loaded_embeddings[:, 1], hue=loaded_categories, alpha=0.6, ax=axes[1])
    axes[1].set_title('With Noise')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')

    plt.tight_layout()
    plt.savefig(f"test_{dataset}")
    if metrics:
        return metric

def loadJson(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def main():
    arg1 = sys.argv[1]
    #arg1 = "pipeline_config.json"
    print("input json:" + arg1)
    #arg2 = sys.argv[2]
    input = loadJson(arg1)
    layers = input["layers"]
    embeddings = input["embedding_data"]
    proven = input["proven_private"]
    comparison = input["comparison"]
    metrics = input["metrics"]
    metrics = Factory(layers, embeddings, proven, comparison, metrics)
    print(metrics)
    


main()