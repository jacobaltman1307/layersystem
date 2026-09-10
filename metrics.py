# Metrics
# https://github.com/hpicgs/topic-models-and-dimensionality-reduction-sensitivity-study
import numpy as np
from scipy import spatial, stats
from sklearn.manifold import trustworthiness
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestCentroid, NearestNeighbors


def get_squared_distances_if_necessary(D_high_l, D_low_l):
    if isinstance(D_high_l, list) or len(D_high_l.shape) == 1:
        D_high = spatial.distance.squareform(D_high_l)
        D_low = spatial.distance.squareform(D_low_l)
    else:
        D_high = D_high_l
        D_low = D_low_l
    return D_high, D_low

def metric_trustworthiness(X_high, X_low, D_high_m=None, D_low_m=None, k=7):
    try:
        return float(trustworthiness(X_high, X_low, n_neighbors=k))
    except (ValueError, TypeError):
        # Invalid input shapes or types => return perfect score as a safe fallback
        return 1.0


def metric_continuity(X_high, X_low, D_high_l=None, D_low_l=None, k=7):
    try:
        return float(trustworthiness(X_low, X_high, n_neighbors=k))
    except (ValueError, TypeError):
        # Invalid input shapes or types => return perfect score as a safe fallback
        return 1.0

def metric_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]

def metric_normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)

def metric_pearson_correlation(D_scatter1, D_scatter2):
    if len(D_scatter1.shape) == 1:
        return stats.pearsonr(D_scatter1, D_scatter2)[0]
    else:
        corrs = []
        for i in range(D_scatter1.shape[1]):
            corrs.append(stats.pearsonr(D_scatter1[:, i], D_scatter2[:, i])[0])
        return np.array(corrs)

def metric_spearman_correlation(D_scatter1, D_scatter2):
    if len(D_scatter1.shape) == 1:
        return stats.spearmanr(D_scatter1, D_scatter2)[0]
    else:
        corrs = []
        for i in range(D_scatter1.shape[1]):
            corrs.append(stats.spearmanr(D_scatter1[:, i], D_scatter2[:, i])[0])
        return np.array(corrs)

def metric_silhouette(X, labels):
    return silhouette_score(X, labels)

def metric_mse(X, X_hat):
    return np.mean(np.square(X - X_hat))

def metric_cluster_ordering(x_low1, x_low2, y):
    clf_scatter1 = NearestCentroid().fit(X=x_low1, y=y)
    clf_scatter2 = NearestCentroid().fit(X=x_low2, y=y)

    distance_list1 = compute_distance_list(clf_scatter1.centroids_)
    distance_list2 = compute_distance_list(clf_scatter2.centroids_)

    return metric_spearman_correlation(distance_list1, distance_list2)

def compute_distance_list(X, eval_distance_metric='euclidean'):
    return spatial.distance.pdist(X, eval_distance_metric)

def metric_absolute_difference_distance_consistency(layout1, layout2, y):
    y_arr = np.asarray(y)
    
    def _distance_consistency(layout):
        clf = NearestCentroid().fit(layout, y_arr)
        return np.mean(clf.predict(layout) == y_arr)

    return abs(_distance_consistency(layout1) - _distance_consistency(layout2))

def metric_spatial_entropy(X, bins=50):
    # Bin the 2D coordinates into a grid
    hist, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    
    # Normalize to get probability distribution
    total_points = np.sum(hist)
    if total_points == 0:
        return 0.0

    p = hist / total_points
    
    # Filter out empty bins to avoid log(0) errors
    p_nonzero = p[p > 0]
    if len(p_nonzero) <= 1:
        return 0.0

    # Calculate Shannon entropy
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    
    # Maximum possible Shannon entropy occurs with a uniform distribution over all bins
    max_entropy = np.log2(hist.size)
    if max_entropy == 0:
        return 0.0

    # Scale from 0 to 1
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def metric_calinski_harabasz(X, labels):
    # Isolate points that actually belong to a cluster
    mask = labels != -1
    X_clustered = X[mask]
    labels_clustered = labels[mask]
    
    # Calinski-Harabasz mathematically requires at least 2 clusters
    if len(np.unique(labels_clustered)) < 2:
        return np.nan
        
    return calinski_harabasz_score(X_clustered, labels_clustered)

def metric_overplotting_penalty(X, epsilon=0.01):
    # Fit KD-Tree on the 2D coordinates
    # k=2 because the first nearest neighbor is the point itself (distance 0.0)
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Isolate the distance to the actual nearest neighbor
    nearest_dist = distances[:, 1]
    
    # Calculate the percentage of points closer than the visual threshold
    penalty_ratio = np.sum(nearest_dist < epsilon) / X.shape[0]
    
    return penalty_ratio

def metric_hopkins_statistic(X, max_samples=1000, random_state=None):
    n_points, dims = X.shape
    
    if n_points < 2:
        return np.nan
        
    M = min(n_points, max_samples)
    rng = np.random.default_rng(random_state)
    
    # 1. Sample M real points from the dataset
    X_sample = rng.choice(X, size=M, replace=False)
    
    # 2. Generate M uniform random points within the 2D bounding box of the dataset
    min_bounds = X.min(axis=0)
    max_bounds = X.max(axis=0)
    uniform_sample = rng.uniform(min_bounds, max_bounds, size=(M, dims))
    
    # 3. Fit KD-Tree on the full dataset for fast lookup
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X)
    
    # 4. Find nearest neighbor distance from uniform points to the dataset
    # (k=1 because the uniform points are not in X)
    dist_uniform, _ = nn.kneighbors(uniform_sample, n_neighbors=1)
    u_squared = np.sum(dist_uniform ** 2)
    
    # 5. Find nearest neighbor distance from the sampled real points to the rest of the dataset
    # (k=2 because the first neighbor is the point itself)
    dist_real, _ = nn.kneighbors(X_sample, n_neighbors=2)
    w_squared = np.sum(dist_real[:, 1] ** 2)
    
    # 6. Calculate Hopkins Statistic
    if (u_squared + w_squared) == 0:
        return 0.5  # Avoid division by zero, return baseline for random distribution
        
    return u_squared / (u_squared + w_squared)


def metric_procrustes(X, Y):
    """
    Compute Procrustes disparity between two embedding layouts X (with noise)
    and Y (without noise). Both must have the same shape (n_samples, n_dimensions).
    Returns disparity in [0.0, 1.0]. Lower values indicate better shape preservation.
    """
    try:
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(Y, dtype=np.float64)
        if x_arr.shape != y_arr.shape or x_arr.shape[0] < 2:
            return 0.0
        _, _, disparity = spatial.procrustes(y_arr, x_arr)
        return float(np.clip(disparity, 0.0, 1.0))
    except Exception:
        return 0.0


def metric_pairwise_distance_kl(X, Y, max_samples=1000, random_state=42):
    """
    Compute the KL divergence between relative pairwise distance probability distributions
    of layout Y (unperturbed, without noise) and layout X (perturbed, with noise):
    D_KL(P_unperturbed || Q_noisy).
    Subsamples to max_samples for consistent cross-run speed and memory bounds.
    Returns non-negative float. Lower values indicate better relative distance preservation.
    """
    try:
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(Y, dtype=np.float64)
        n = x_arr.shape[0]
        if n < 2 or x_arr.shape != y_arr.shape:
            return 0.0

        M = min(n, max_samples)
        if M < n:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=M, replace=False)
            x_sub = x_arr[idx]
            y_sub = y_arr[idx]
        else:
            x_sub = x_arr
            y_sub = y_arr

        d_y = spatial.distance.pdist(y_sub, metric='euclidean')
        d_x = spatial.distance.pdist(x_sub, metric='euclidean')

        sum_y = np.sum(d_y)
        sum_x = np.sum(d_x)
        if sum_y == 0 or sum_x == 0:
            return 0.0

        P = d_y / sum_y
        Q = d_x / sum_x

        # Stabilize with small epsilon to prevent log(0) or div by 0
        eps = 1e-12
        P = np.clip(P, eps, 1.0)
        P /= np.sum(P)
        Q = np.clip(Q, eps, 1.0)
        Q /= np.sum(Q)

        kl = stats.entropy(P, Q)
        return float(max(0.0, kl))
    except Exception:
        return 0.0


def compute_scaled_human_utility(metrics_dict):
    """
    Compute Estimated Human Utility as the average of polarity-aligned,
    min-max normalized human visual utility sub-metrics:
      - Benefit metrics (higher is better):
          'dbscan_clusters', 'spatial_entropy', 'hopkins_statistic'
      - Cost / Penalty metrics (lower is better, inverted so higher = better):
          'overplotting_penalty', 'absolute_difference'

    Each available 2D sub-metric grid is scaled to [0.0, 1.0].
    Returns an array of the same shape with values strictly in [0.0, 1.0],
    or None if no sub-metrics are found.
    """
    if not metrics_dict:
        return None

    benefit_keys = ("dbscan_clusters", "spatial_entropy", "hopkins_statistic")
    penalty_keys = ("overplotting_penalty", "absolute_difference")

    scaled_grids = []

    for k in benefit_keys:
        if k in metrics_dict and metrics_dict[k] is not None:
            arr = np.asarray(metrics_dict[k], dtype=np.float64)
            if arr.size == 0:
                continue
            min_v = np.nanmin(arr)
            max_v = np.nanmax(arr)
            if np.isnan(min_v) or np.isnan(max_v):
                continue
            if max_v > min_v:
                s = (arr - min_v) / (max_v - min_v)
            else:
                s = np.ones_like(arr)
            scaled_grids.append(s)

    for k in penalty_keys:
        if k in metrics_dict and metrics_dict[k] is not None:
            arr = np.asarray(metrics_dict[k], dtype=np.float64)
            if arr.size == 0:
                continue
            min_v = np.nanmin(arr)
            max_v = np.nanmax(arr)
            if np.isnan(min_v) or np.isnan(max_v):
                continue
            if max_v > min_v:
                s = (max_v - arr) / (max_v - min_v)
            else:
                s = np.ones_like(arr)
            scaled_grids.append(s)

    if not scaled_grids:
        return None

    composite = np.nanmean(np.stack(scaled_grids, axis=0), axis=0)
    return np.clip(composite, 0.0, 1.0)