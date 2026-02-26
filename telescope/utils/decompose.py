# -*- coding: utf-8 -*-

# This file is part of TelescopeGL.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Connected component block decomposition for parallel EM.

Decomposes the N x K score matrix into independent blocks using
scipy.sparse.csgraph.connected_components on the feature adjacency graph.
"""

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components


def find_blocks(score_matrix):
    """Find independent blocks via connected components of the feature graph.

    Builds a K x K feature adjacency matrix: two features are connected if
    they share at least one fragment (row with nonzero in both columns).

    Args:
        score_matrix: N x K sparse CSR matrix of alignment scores.

    Returns:
        (n_components, labels): number of blocks and per-feature label array.
    """
    # Binary indicator: which cells are nonzero
    binary = (score_matrix > 0).astype(np.float64)
    # K x K adjacency: features sharing fragments
    adj = binary.T @ binary
    n_components, labels = connected_components(adj, directed=False)
    return n_components, labels


def split_matrix(score_matrix, labels, n_components):
    """Split the score matrix into independent sub-problems.

    Args:
        score_matrix: N x K sparse CSR matrix.
        labels: Per-feature label array from find_blocks().
        n_components: Number of connected components.

    Returns:
        List of (sub_matrix, feat_indices, row_indices) tuples.
    """
    blocks = []
    for c in range(n_components):
        feat_indices = np.where(labels == c)[0]
        # Extract columns for this block
        sub = score_matrix[:, feat_indices]
        # Find rows with any nonzero in these columns
        row_indices = sub.getnnz(axis=1).nonzero()[0]
        sub = sub[row_indices, :]
        # Convert to CSR (column slicing may produce CSC)
        sub = scipy.sparse.csr_matrix(sub)
        blocks.append((sub, feat_indices, row_indices))
    return blocks


def merge_results(block_results, blocks_info, K):
    """Reassemble full-size pi, theta, z, and lnl from block results.

    Args:
        block_results: List of (pi_block, theta_block, z_block, lnl_block) tuples.
        blocks_info: List of (sub_matrix, feat_indices, row_indices) from split_matrix().
        K: Total number of features in the original matrix.

    Returns:
        (pi, theta, z_rows, z_cols, z_data, lnl): merged results.
            z is returned as COO components for efficient CSR construction.
    """
    pi = np.zeros(K, dtype=np.float64)
    theta = np.zeros(K, dtype=np.float64)
    total_lnl = 0.0

    z_rows = []
    z_cols = []
    z_data = []

    for (pi_b, theta_b, z_b, lnl_b), (_, feat_indices, row_indices) in zip(
            block_results, blocks_info):
        # Map block parameters back to global indices
        pi[feat_indices] = pi_b
        theta[feat_indices] = theta_b
        total_lnl += lnl_b

        # Map block z to global coordinates
        z_coo = scipy.sparse.coo_matrix(z_b)
        global_rows = row_indices[z_coo.row]
        global_cols = feat_indices[z_coo.col]
        z_rows.append(global_rows)
        z_cols.append(global_cols)
        z_data.append(z_coo.data)

    z_rows = np.concatenate(z_rows) if z_rows else np.array([], dtype=np.intp)
    z_cols = np.concatenate(z_cols) if z_cols else np.array([], dtype=np.intp)
    z_data = np.concatenate(z_data) if z_data else np.array([], dtype=np.float64)

    # Renormalize pi to sum to 1 across all blocks
    pi_sum = pi.sum()
    if pi_sum > 0:
        pi /= pi_sum

    return pi, theta, z_rows, z_cols, z_data, total_lnl
