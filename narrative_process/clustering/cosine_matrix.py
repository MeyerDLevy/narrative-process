# === clustering/cosine_matrix.py ===
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.linalg import inv
import datetime

def embeddingsmatrix2cosinesimmat(matrix, distancemetric='cosine', batch_size=10000, dtype=np.float16, memmap_file=''):
    start_time = datetime.datetime.now()
    n = matrix.shape[0]

    # Try to create or overwrite the memmap file
    try:
        if os.path.exists(memmap_file):
            os.remove(memmap_file)
        distance_matrix = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=(n, n))
    except PermissionError:
        alt_file = memmap_file.replace('.mmap', '_alt.mmap')
        print(f"[Warning] File in use, writing to alternate memmap: {alt_file}")
        memmap_file = alt_file
        distance_matrix = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=(n, n))

    if distancemetric == 'mahalanobis':
        covariance_matrix = np.cov(matrix, rowvar=False)
        inv_cov_matrix = inv(covariance_matrix)

    for i in tqdm(range(0, n, batch_size), desc='Processing rows'):
        end_i = min(i + batch_size, n)
        batch_i = matrix[i:end_i]
        for j in range(i, n, batch_size):
            end_j = min(j + batch_size, n)
            batch_j = matrix[j:end_j]

            if distancemetric == 'cosine':
                block = cosine_similarity(batch_i, batch_j)
            elif distancemetric == 'euclidean':
                block = euclidean_distances(batch_i, batch_j)
            elif distancemetric == 'manhattan':
                block = manhattan_distances(batch_i, batch_j)
            elif distancemetric == 'mahalanobis':
                block = np.array([
                    [np.sqrt((p - q).dot(inv_cov_matrix).dot((p - q).T)) for q in batch_j]
                    for p in batch_i
                ])
            else:
                raise ValueError("Unsupported distance metric.")

            distance_matrix[i:end_i, j:end_j] = block.astype(dtype)
            if j != i:
                distance_matrix[j:end_j, i:end_i] = block.T.astype(dtype)

    del distance_matrix  # Close and flush to disk
    print(f"[{datetime.datetime.now()}] Cosine matrix saved to {memmap_file}")
    return memmap_file


def build_network_with_labels(memmap_matrix, keys, threshold=0.33, chunk_size=1000):
    import networkx as nx
    G = nx.Graph()
    keys_list = list(keys)
    n_rows, n_cols = memmap_matrix.shape
    for chunk_start in tqdm(range(0, n_rows, chunk_size), desc='Building Network'):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        chunk = memmap_matrix[chunk_start:chunk_end, :]
        for i in tqdm(range(chunk.shape[0])):
            for j in range(n_cols):
                data = chunk[i, j]
                if data > threshold:
                    row_label = keys_list[chunk_start + i]
                    col_label = keys_list[j]
                    G.add_edge(row_label, col_label, weight=data)
    return G

