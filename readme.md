
# Layer based system for differential privacy and dimensionality reduction

## Goal: Find out when is the best time to add noise for differential privacy to document embeddings



![alt text](examples/example_plot.png)


## Setup

Requirements: 
    
    - 16GB+ of avaliable system memory
    - GPU for embedding
    - CPU time
    Python 3.10+ virutal environment
`pip install -r requirements.txt`

## Supported Datasets

| Dataset | Classes | Source |
|---------|---------|--------|
| `emails` (default) | Crime, Entertainment, Politics, Science | Local `cleanedData.csv` |
| `agnews` | World, Sports, Business, Sci/Tech | HuggingFace `ag_news` (120k train) |
| `yelp` | 1–5 stars | HuggingFace `yelp_review_full` (40k subset) |

## Steps

1. Clean the original email data with `preprocess/clean.py` to get rid of empty datapoints

2. **Download additional datasets** (AG News and Yelp):
    ```
    python preprocess/download_datasets.py               # downloads both
    python preprocess/download_datasets.py --dataset agnews   # AG News only
    python preprocess/download_datasets.py --dataset yelp     # Yelp only
    python preprocess/download_datasets.py --yelp-subset-size 10000  # custom subset size
    ```

3. Run one of the scripts in `embedding/` to save the documents as embeddings, using the `--dataset` flag:
    ```
    python embedding/save_bert.py --dataset emails   # original dataset (default)
    python embedding/save_bert.py --dataset agnews   # AG News
    python embedding/save_bert.py --dataset yelp     # Yelp
    ```
    There are 5 embedding model options provided: bert, gemma, gemma3, qwen8b, vault. All support the `--dataset` flag.

4. Run the dimensionality reduction gridsearch with the `--dataset` flag (or auto-detected from the embedding file):
    ```
    python gridsearch.py -f embeddingdatabert-base-nli-mean-tokens_emails.npz -r 1 -t PCA -s 0
    python gridsearch.py -f embeddingdatabert-base-nli-mean-tokens_agnews.npz -r 1 -t PCA -s 0
    python gridsearch.py -f embeddingdatabert-base-nli-mean-tokens_yelp.npz -r 1 -t PCA -s 0
    ```
    - `-f`, `--embedding-file` Path to the embedding file
    - `-r`, `--runs` Number of runs to execute
    - `--run` used to run a specific run (for HPC systems or redoing specific runs)
    - `-t`, `--dr-type` The primary type of dimensionality reduction (DR) that happens in the first layer
    - `-t2` `--dr-type-secondary` The secondary type of DR that occurs after noise is added (default is the same as primary)
    - `-s` `--resolution` Changes the grid size resolution in a range of 0-4
    - `--no-plot` Removes the automatic plotting from the gridsearch script
    - `--dataset` Explicitly set dataset name ('agnews', 'emails', 'yelp'; auto-detected from .npz if omitted)

    Results are saved to `runs/<dataset>/<embeddingModel>/<dimReductType>/`


5. Plot results with: `python plot_results.py -c runs/` 
    - Option 1: `-c` `--crawl` Crawls through a directory recursively to find all output files
    - Option 2: `-d` `--directory` Choose one directory to make into plots
    - Option 3: `-f` `--files` Choose files to add manually
    - `-o` `--output` Change output file name
    - `-ao` `--average-ouput` Change the average plot's file name
    - `-hu` `--human-utility` Generate human utility sister plots

6. Compile results across all algorithms and identify best methods per grid cell with: `python compile_results.py 
    - Option 1: `python compile_results.py runs/` (or `-c runs/`) Crawls `runs/` and compiles all algorithms per dataset/embedding
    - `-hu`, `--human-utility`: Also compile and plot best methods for Estimated Human Visual Utility
    - `-ss`, `--skip-supervised`: Skip compilation for supervised algorithms (e.g., LDA)
    - `-m`, `--metric`: Target a specific individual metric (e.g. `silhouette`, `trustworthiness`)
    - `-o`, `--output-dir`: Change base output directory (default: `output/compiled/`)

7. Export cell-by-cell raw metric data across all runs to an analysis-ready CSV with: `python export_raw_metrics.py runs/`
    - Outputs `output/compiled/raw_metrics_tidy.csv` with mean and standard deviation for each metric across runs.
    - `-o`, `--output`: Custom CSV destination path
    - `--no-std`: Omit standard deviation columns (mean only)
    - `--export-matrices`: Also dump individual 2D matrix CSVs per metric to `output/compiled/raw_matrices/`
    - `-d`, `--dataset`: Filter by specific dataset(s)
    - `-e`, `--embedding`: Filter by specific embedding model(s)
    - `-m`, `--method`: Filter by specific dimensionality reduction method(s)
    - `--metrics`: Filter specific metric columns
    - `-ss`, `--skip-supervised`: Skip supervised dimensionality reduction algorithms (e.g. LDA)
    *(Note: `compile_results.py` also exports `output/compiled/raw_metrics_tidy.csv` automatically).*

8. Compare graphs and find the best balance for the required epsilon

9. Make/edit a json layer file (examples in `examples/`) then run `python Factory.py path_to_json`
    - The layers are executed from top to bottom and there can be as many layers as needed.

## Output Directory Structure

```
runs/
  <dataset>/           # e.g., emails, agnews, yelp
    <embedding_model>/
      <dim_reduct_type>/

        gridsearch_results_*.npz
        gridsearch_comparison.png
        gridsearch_comparison_average.png
```

## Dimensionality Reduction types:

    - PCA 
    - TSNE
    - LDA
    - SVD
    - MDS
    - LLE
    - SOM
    - UMAP

## Metrics

### Dimensionality Reduction Quality & Sensitivity
- **Continuity**: Neighborhood preservation from low to high dimensional space.
- **Trustworthiness**: Neighborhood preservation from high to low dimensional space.
- **Cluster Ordering**: Consistency of cluster centroid rank distances.
- **Pearson Correlation**: Linear correlation of distance matrices.
- **Spearman Correlation**: Monotonic correlation of distance matrices.
- **Silhouette Score**: Cluster tightness and separation.
- **Procrustes Disparity** (`metric_procrustes`): Rigid shape disparity $\in [0, 1]$ between perturbed (with noise) and unperturbed (without noise) projections after optimal translation, rotation, and uniform scaling.
- **Pairwise Distance Distribution KL Divergence** (`metric_pairwise_distance_kl`): Relative distance distribution divergence $D_{KL}(P_{\text{unperturbed}} \parallel Q_{\text{noisy}})$ measuring deformation of inter-point distance structures.
- **Average Metrics**: Composite mean score of core rank/correlation metrics.
- **CPU Process Time**: Total execution wall-clock time in seconds.

### Human Visual Utility Metrics
- **DBSCAN # Clusters**: Estimated number of visual clusters discovered.
- **Spatial/Image Entropy**: Uniformity vs structure of spatial layout.
- **Overplotting / Crowding Penalty**: Ratio of points crowded within visual distance threshold $\epsilon$.
- **Hopkins Statistic**: Measure of clustering tendency against spatial randomness.
- **Absolute Difference Distance Consistency**: Mean absolute discrepancy in pairwise distances.
- **Estimated Human Utility**: Aggregated visual utility score.

## Resolutions: Epsilons | output dimensions

    - 0 [1,2] | [768,2]
    - 1 [1,10,50,100,500,1000,1000000000] | [768,3,2]
    - 2 [1,10,50,100,500,1000] | [768,384,128,48,8,2]
    - 3 [1,10,50,100,500,1000,5000,10000,1000000000] | [768,512,256,128,64,32,16,8,4,2]
    - 4 [.1,.5,1,5,10,25,50,100,250,500,1000,2500,5000,10000,1000000000] | [768,512,256,128,96,64,32,16,12,8,6,4,3,2]
    - 5 [1,10,50,100,500,1000,5000,10000,1000000000] | [categories-1, ..., 2] (adaptive for algorithms like LDA and TSNE)
