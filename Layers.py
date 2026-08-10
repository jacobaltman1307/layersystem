import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn_som.som import SOM
import warnings
from scipy.sparse import SparseEfficiencyWarning

# Suppress scipy sparse efficiency warnings caused by sklearn's Isomap
warnings.simplefilter("ignore", category=SparseEfficiencyWarning)


#
def ReductionLayer(X, y, Type, outputDim, catagories):
    out = DimReduction(X, y, Type, outputDim, catagories)
    return out

def NoiseLayer(X, epsilon, proven):
    out = perturbAll(X, epsilon, proven)
    return out




#Dim reduction
def DimReduction(X, y, t, outputDim, catagories):
    X_processed = StandardScaler().fit_transform(X)
    out = None
    #print(t)
    match t:
        case "PCA":
            pca = PCA(n_components=outputDim)
            out = pca.fit_transform(X_processed)
        case "TSNE":
            tsne = TSNE(n_components=outputDim, random_state=42)
            out = tsne.fit_transform(X_processed)
        case "LDA":
            #print(catagories)
            if outputDim > (len(catagories) - 1):
                raise ValueError(f"The output Dim for LDA cannot be more than components - 1. max: {len(catagories) - 1} vs your: {outputDim}")
            lda = LinearDiscriminantAnalysis(n_components=outputDim)
            out = lda.fit_transform(X_processed,y)
        case "SVD":
            svd = TruncatedSVD(n_components=outputDim)
            out = svd.fit_transform(X_processed)
        case "ISOMAP":
            isomap = Isomap(n_components=outputDim, n_neighbors=15)
            out = isomap.fit_transform(X_processed)
        case "MDS":
            X_proc_clean = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0)
            from sklearn.metrics import euclidean_distances
            D = euclidean_distances(X_proc_clean)
            D = (D + D.T) / 2.0
            mds = MDS(n_components=outputDim, random_state=42, n_init=2, max_iter=25, dissimilarity='precomputed', init='random')
            out = mds.fit_transform(D)
        case "LLE":
            lle = LocallyLinearEmbedding(n_components=outputDim, random_state=42, eigen_solver="dense")
            out = lle.fit_transform(X_processed)
        case "SOM":
            som = SOM(m=outputDim, n=1, dim=X.shape[1])
            som.fit(X_processed)
            out = som.transform(X_processed)
        case "latentDA":
            X_abs = np.abs(X_processed)
            lda = LatentDirichletAllocation(n_components=outputDim)
            lda.fit(X_abs)
            out = lda.transform(X_abs)
        case "UMAP":
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=outputDim)
            out = reducer.fit_transform(X)
        case _:
            print("No Dim reduction found")
    
    return out



#Noise
def get_perturbed_vector(epsilon, word_vec, n):
    noise = np.random.normal(0, 1, n)
    norm_noise = noise / np.linalg.norm(noise)
    N = np.random.gamma(n, 1/epsilon) * norm_noise
    return word_vec + N, N

def proven_get_perturbed_vector(epsilon, word_vec, n):
    noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
    norm_noise = noise / np.linalg.norm(noise)
    N = np.random.gamma(n, 1/epsilon) * norm_noise
    return word_vec + N, N

def perturbAll(X, epsilon, proven):
    X_perturbed = X.copy()
    if proven:
        for i in range(len(X_perturbed)):
            X_perturbed[i], _ = get_perturbed_vector(epsilon, X_perturbed[i], len(X_perturbed[i]))
    else:
        for i in range(len(X_perturbed)):
            X_perturbed[i], _ = get_perturbed_vector(epsilon, X_perturbed[i], len(X_perturbed[i]))
    return X_perturbed


