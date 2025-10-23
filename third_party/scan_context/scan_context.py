# Code based on ScanContext implementation: https://github.com/irapkaist/scancontext/blob/master/python/make_sc_example.py
# Optimized implementation by Jacek Komorowski

import numpy as np
from sklearn.neighbors import KDTree


class ScanContext:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        return x


def distance_sc(sc1: np.ndarray, sc2: np.ndarray):
    # Distance between 2 scan context descriptors
    assert sc1.shape == sc2.shape
    num_sectors = sc1.shape[1]

    # Repeate to move 1 columns
    _one_step = 1  # const
    sim_for_each_cols = np.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = np.roll(sc1, _one_step, axis=1)  # columne shift

        # Compare
        sc1_norm = np.linalg.norm(sc1, axis=0)
        sc2_norm = np.linalg.norm(sc2, axis=0)
        mask = ~np.logical_or(np.isclose(sc1_norm, 0.), np.isclose(sc2_norm, 0.))

        # Compute cosine similarity between columns of sc1 and sc2
        cossim = np.sum(np.multiply(sc1[:, mask], sc2[:, mask]), axis=0) / (sc1_norm[mask] * sc2_norm[mask])

        sim_for_each_cols[i] = np.sum(cossim) / np.sum(mask)

    yaw_diff = (np.argmax(sim_for_each_cols) + 1) % sc1.shape[1]  # because python starts with 0
    sim = np.max(sim_for_each_cols)
    dist = 1. - sim

    return dist, yaw_diff


def sc2rk(sc):
    # Scan context to ring key
    return np.mean(sc, axis=1)


class ScanContextManager:
    def __init__(self, max_capacity=100000):
        # max_capacity: maximum number of nodes
        self.num_sector = None
        self.num_ring = None
        self.max_capacity = max_capacity

        self.sc = ScanContext()
        self.scancontexts = None
        self.ringkeys = None
        self.curr_node_idx = 0
        self.knn_tree_idx = -1      # Number of nodes for which KDTree tree was computed (recalculate KDTree only after new nodes are added)
        self.ringkey_tree = None

    def add_node(self, image):
        # Compute and store the descriptor
        # pc: (N, 3) point cloud
        assert image.ndim == 2

        if self.scancontexts is None:
            self.num_ring = image.shape[0]
            self.num_sector = image.shape[1]
            self.scancontexts = np.zeros((self.max_capacity, self.num_ring, self.num_sector))
            self.ringkeys = np.zeros((self.max_capacity, self.num_ring))

        # Compute
        sc = self.sc(image)        # (num_ring, num_sector) array
        rk = sc2rk(image)          # (num_ring,) array

        self.scancontexts[self.curr_node_idx] = sc
        self.ringkeys[self.curr_node_idx] = rk
        self.curr_node_idx += 1
        assert self.curr_node_idx < self.max_capacity, f'Maximum ScanContextManager capacity exceeded: {self.max_capacity}'

    def query(self, query_pc, k=1, reranking=True):
        assert self.curr_node_idx > 0, 'Empty database'

        if self.curr_node_idx != self.knn_tree_idx:
            # recalculate KDTree
            self.ringkey_tree = KDTree(self.ringkeys[:self.curr_node_idx-1])
            self.knn_tree_idx = self.curr_node_idx

        # find candidates
        query_sc = self.sc(query_pc)        # (num_ring, num_sector) array
        query_rk = sc2rk(query_sc)          # (num_ring,) array
        # KDTree query expects (n_samples, sample_dimension) array
        _, nn_ndx = self.ringkey_tree.query(query_rk.reshape(1, -1), k=k)
        # nncandidates_idx is (n_samples=1, sample_dimension) array
        nn_ndx = nn_ndx[0]

        # step 2
        sc_dist = np.zeros((k,))
        sc_yaw_diff = np.zeros((k,))

        if not reranking:
            return nn_ndx, None, None

        # Reranking using the full ScanContext descriptor
        for i in range(k):
            candidate_ndx = nn_ndx[i]
            candidate_sc = self.scancontexts[candidate_ndx]
            sc_dist[i], sc_yaw_diff[i] = distance_sc(candidate_sc, query_sc)

        reranking_order = np.argsort(sc_dist)
        nn_ndx = nn_ndx[reranking_order]
        sc_yaw_diff = sc_yaw_diff[reranking_order]
        sc_dist = sc_dist[reranking_order]

        return nn_ndx, sc_dist, sc_yaw_diff
