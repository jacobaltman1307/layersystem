#!/bin/bash
#SBATCH --job-name=gridsearch_isomap
#SBATCH --output=outputs/isomap_%A_%a.log
#SBATCH --error=outputs/isomap_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-179%30         # 36 configs * 5 runs = 180 tasks (max 30 concurrent)
#SBATCH --partition=Orion

# ==============================================================================
# Multi-threaded environment setup
# ==============================================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load module and activate virtual environment
module load python/3.12.3
source $SLURM_SUBMIT_DIR/layer_env/bin/activate

mkdir -p outputs

# ==============================================================================
# 36 ISOMAP Configurations (12 embedding files * 3 method variants)
# ==============================================================================
CONFIGS=(
    # --- Vaultgemma-1b ---
    "-f embeddingdataVaultgemma-1b_agnews.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_agnews.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_agnews.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_yelp.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_yelp.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_yelp.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_emails.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_emails.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdataVaultgemma-1b_emails.npz -t ISOMAP -s 3 -m 25000"

    # --- bert-base-nli-mean-tokens ---
    "-f embeddingdatabert-base-nli-mean-tokens_agnews.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens_agnews.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens_agnews.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens_yelp.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens_yelp.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens_yelp.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdatabert-base-nli-mean-tokens.npz -t ISOMAP -s 3 -m 25000"

    # --- embeddinggemma-300m ---
    "-f embeddingdataembeddinggemma-300m_agnews.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m_agnews.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m_agnews.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m_yelp.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m_yelp.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdataembeddinggemma-300m_yelp.npz -t ISOMAP -s 3 -m 25000"

    # --- gemma-3-1b-pt ---
    "-f embeddingdatagemma-3-1b-pt_agnews.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_agnews.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_agnews.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_yelp.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_yelp.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_yelp.npz -t ISOMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_emails.npz -t UMAP -t2 ISOMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_emails.npz -t ISOMAP -t2 UMAP -s 3 -m 25000"
    "-f embeddingdatagemma-3-1b-pt_emails.npz -t ISOMAP -s 3 -m 25000"
)

# ==============================================================================
# Map SLURM_ARRAY_TASK_ID to (Config Index, Run Index)
# Each SLURM task executes exactly ONE run
# ==============================================================================
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
RUN_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

CURRENT_CONFIG=${CONFIGS[$CONFIG_IDX]}

echo "========================================================================"
echo "Job Array ID : $SLURM_ARRAY_JOB_ID, Task ID : $SLURM_ARRAY_TASK_ID"
echo "Config Index : $CONFIG_IDX / ${#CONFIGS[@]}"
echo "Run Index    : $RUN_IDX (out of 5 runs: 0..4)"
echo "Config Args  : $CURRENT_CONFIG"
echo "Start Time   : $(date)"
echo "Host Node    : $(hostname)"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "========================================================================"

# Run single run with --skip-existing and --no-plot
python gridsearch.py $CURRENT_CONFIG --run $RUN_IDX --no-plot --skip-existing

EXIT_CODE=$?
echo "========================================================================"
echo "Exit Code    : $EXIT_CODE"
echo "End Time     : $(date)"
echo "========================================================================"
exit $EXIT_CODE
