import numpy as np
import heapq
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from narrative_process.clustering.cosine_matrix import embeddingsmatrix2cosinesimmat

def loadmmap(mmfilepath, shape, dtype="float16"):
    return np.memmap(mmfilepath, dtype=dtype, shape=shape, mode='r+')

def replace_indices_with_names(clusters, names):
    #return {names[key]: [names[idx] for idx in indices] for key, indices in clusters.items()}
    #return {key: [names[idx] for idx in indices] for key, indices in clusters.items()}
    return {key: [names[arg] for arg in args] for key, args in clusters.items()}

def agglomerative_clustering(embeddings, names, mmfile, threshold=0.9, maxclusterprop=0.05):
    num_samples = embeddings.shape[0]
    maxclustersize = int(np.round(num_samples * maxclusterprop))
    clusters = {i: [i] for i in range(num_samples)}
    cluster_embeddings = {i: embeddings[i] for i in range(num_samples)}
    cluster_indices = list(range(num_samples))

    esm = embeddingsmatrix2cosinesimmat(embeddings, memmap_file=mmfile)
    mm = loadmmap(esm, (num_samples, num_samples))

    def similarity_pairs():
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                yield (-mm[i, j], i, j)

    pq = []
    gen = similarity_pairs()

    for _ in range(100000):  # Initial heap fill
        try:
            heapq.heappush(pq, next(gen))
        except StopIteration:
            break

    pbar = tqdm(total=num_samples, desc='Agglomerative Clustering')
    clusterstoobig = set()

    while pq:
        max_sim, i, j = heapq.heappop(pq)
        max_sim = -max_sim
        if max_sim < threshold:
            break
        if i in clusters and j in clusters and i not in clusterstoobig:
            if len(clusters[i]) >= maxclustersize or len(clusters[i]) + len(clusters[j]) > maxclustersize:
                clusterstoobig.add(i)
                continue
            clusters[i].extend(clusters[j])
            del clusters[j]

            cluster_embeddings[i] = np.mean([cluster_embeddings[i], cluster_embeddings[j]], axis=0)
            del cluster_embeddings[j]

        try:
            for _ in range(1000):  # Refill heap incrementally
                heapq.heappush(pq, next(gen))
        except StopIteration:
            pass

        pbar.update(1)

    return replace_indices_with_names(clusters, names)
