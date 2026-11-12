import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

def directed(pairs):
    p = {tuple(sorted((int(i), int(j)))) for i, j in pairs if i != j}
    return np.array(sorted([(i, j) for i, j in p] + [(j, i) for i, j in p]), dtype=np.int64).reshape(-1, 2).T

def build_hierarchy(x, coordinates, barcodes, pseudo_coordinates=None, seed=3927):
    n = len(x)
    canonical = np.argsort(np.asarray(barcodes, dtype=str), kind='stable')
    edges = {}
    members = {}
    representatives = {}
    if n > 1:
        a = kneighbors_graph(coordinates, n_neighbors=min(8, n - 1), mode='connectivity').tocoo()
        edges['spatial_local'] = directed(zip(a.row, a.col))
    else:
        edges['spatial_local'] = np.empty((2, 0), dtype=np.int64)
    for view, values in [('spatial', coordinates), ('morphology', x)]:
        labels_sorted = KMeans(min(5, n), n_init=10, max_iter=1000, random_state=seed).fit_predict(values[canonical])
        labels = np.empty(n, dtype=np.int64)
        labels[canonical] = labels_sorted
        stars = []
        reps = []
        for g in np.unique(labels):
            ids = canonical[labels[canonical] == g]
            centroid = x[ids].mean(0)
            rep = int(ids[np.argmin(np.linalg.norm(x[ids] - centroid, axis=1))])
            stars.extend(((int(i), rep) for i in ids if i != rep))
            reps.append(rep)
        members[view] = np.eye(5, dtype=np.float32)[labels]
        representatives[view] = reps
        edges[view + '_cluster_star'] = directed(stars)
        edges[view + '_representative_shortcut'] = directed(((i, j) for i in reps for j in reps))
    edges['cross_representative_shortcut'] = directed(((i, j) for i in representatives['spatial'] for j in representatives['morphology']))
    pseudo = np.empty((0, 2), dtype=np.float32) if pseudo_coordinates is None else pseudo_coordinates
    if len(pseudo):
        nn = NearestNeighbors(n_neighbors=min(8, n)).fit(coordinates).kneighbors(pseudo, return_distance=False)
        edges['pseudo_context'] = directed(((n + i, int(j)) for i, row in enumerate(nn) for j in row))
    else:
        edges['pseudo_context'] = np.empty((2, 0), dtype=np.int64)
    return dict(edges=edges, spatial_membership=members['spatial'], morphology_membership=members['morphology'], representatives=representatives)
