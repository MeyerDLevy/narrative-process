# === clustering/community.py ===
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import community as community_louvain

def part2clusterdict(partition):
    clusters = defaultdict(list)
    for node, cluster_id in partition.items():
        clusters[cluster_id].append(node)
    return clusters

def dynamicclustersameenough(cluster, mat, names2mminddict, base_thresh=0.80, min_thresh=0.65, dynamic_size=15):
    inds = [names2mminddict[item] for item in cluster]
    cluster_size = len(cluster)
    if cluster_size < dynamic_size:
        dynamic_thresh = base_thresh - (base_thresh - min_thresh) * (1 - (cluster_size / dynamic_size))
        threshold = max(dynamic_thresh, min_thresh)
    else:
        threshold = base_thresh
    if len(inds) > 1:
        similarities = [mat[x, y] for x in inds for y in inds if x != y]
    else:
        return True
    if not similarities:
        return True
    return np.mean(similarities) >= threshold

def buildtermgraph(ctokens, names2mminddict, mm):
    cmminds = [names2mminddict[tok] for tok in ctokens]
    g = nx.Graph()
    for xi, x in enumerate(ctokens):
        for yi, y in enumerate(ctokens):
            if x != y:
                wval = mm[cmminds[xi], cmminds[yi]]
                g.add_edge(x, y, weight=max(wval, 0.0))
    return g

def analyze_clusters(clusters, mm, names2mminddict,
                     clustersimthresh=0.85, clustersimminthresh=0.5,
                     resbase=1.0, scaleratchet=0.025):
    itercount = 0
    clusters_to_check = list(clusters.keys())
    while True:
        cluster_checks = [(ckey, dynamicclustersameenough(clusters[ckey], mm, names2mminddict,
                                        base_thresh=clustersimthresh,
                                        min_thresh=clustersimminthresh))
                          for ckey in tqdm(clusters_to_check)]
        testtruetotal = sum(result for _, result in cluster_checks)
        itercount += 1
        if itercount > 1 and testtruetotal == len(clusters_to_check):
            break
        new_clusters_to_check = []
        for ckey, result in cluster_checks:
            if not result:
                ctokens = clusters[ckey]
                smallg = buildtermgraph(ctokens, names2mminddict, mm)
                smallpartition = community_louvain.best_partition(smallg, weight='weight', resolution=resbase + scaleratchet)
                smallclusters = {f"{ckey}_{key}": value for key, value in part2clusterdict(smallpartition).items()}
                clusters.update(smallclusters)
                del clusters[ckey]
                new_clusters_to_check.extend(smallclusters.keys())
        clusters_to_check = new_clusters_to_check
        if not clusters_to_check:
            break
    return clusters

def get_most_central_terms(clusters, names2mminddict, mm):
    rowdicts = []
    for ckey in tqdm(clusters.keys()):
        terms = clusters[ckey]
        if len(terms) > 1:
            clusterg = buildtermgraph(terms, names2mminddict, mm)
            try:
                mostcentralterm = pd.Series(nx.eigenvector_centrality(clusterg, weight="weight")).idxmax()
            except:
                mostcentralterm = pd.Series(terms).apply(len).idxmax()
        else:
            mostcentralterm = terms[0]
        rowdicts.append({"key": ckey, "mostcentralterm": mostcentralterm})
    return pd.DataFrame(rowdicts)
