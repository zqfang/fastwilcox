"""Ranking functions for matrices with tie handling."""

import numpy as np
from scipy import sparse
from typing import List, Tuple


def in_place_rank_mean(v: np.ndarray, idx_begin: int, idx_end: int) -> List[float]:
    """Calculate ranks with mean ties in-place.

    Args:
        v: Array to rank (will be modified in-place)
        idx_begin: Start index
        idx_end: End index (inclusive)

    Returns:
        List of tie group sizes
    """
    ties = []

    if idx_begin > idx_end:
        return ties

    # Create (value, original_index) pairs
    n = idx_end - idx_begin + 1
    v_sort = [(v[idx_begin + i], i) for i in range(n)]

    # Sort by value
    v_sort.sort(key=lambda x: x[0])

    rank_sum = 0.0
    tie_count = 1.0
    i = 1

    while i < len(v_sort):
        if v_sort[i][0] != v_sort[i - 1][0]:
            # Current value differs from previous - assign ranks to previous group
            for j in range(int(tie_count)):
                orig_idx = v_sort[i - 1 - j][1]
                v[orig_idx + idx_begin] = (rank_sum / tie_count) + 1.0

            # Reset counters
            rank_sum = float(i)
            if tie_count > 1.0:
                ties.append(tie_count)
            tie_count = 1.0
        else:
            # Current value is tied - accumulate rank
            rank_sum += float(i)
            tie_count += 1.0

        i += 1

    # Process the last element(s)
    for j in range(int(tie_count)):
        orig_idx = v_sort[i - 1 - j][1]
        v[orig_idx + idx_begin] = (rank_sum / tie_count) + 1.0

    if tie_count > 1.0:
        ties.append(tie_count)

    return ties


def rank_matrix_dense(x: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
    """Rank dense matrix column-wise.

    Args:
        x: Matrix to rank (nrows × ncols)

    Returns:
        Tuple of (ranked_matrix, ties) where ties is a list of tie counts per column
    """
    nrows, ncols = x.shape
    x_ranked = x.copy()
    ties = [[] for _ in range(ncols)]

    for c in range(ncols):
        # Create (value, index) pairs for sorting
        v_sort = [(x[i, c], i) for i in range(nrows)]

        # Sort by value
        v_sort.sort(key=lambda x: x[0])

        rank_sum = 0.0
        tie_count = 1
        i = 1

        # Process all but the last element
        while i < len(v_sort):
            if v_sort[i][0] != v_sort[i - 1][0]:
                # Current value differs from previous - assign ranks to previous group
                for j in range(tie_count):
                    idx = v_sort[i - 1 - j][1]
                    x_ranked[idx, c] = (rank_sum / tie_count) + 1.0

                # Reset counters
                rank_sum = float(i)
                if tie_count > 1:
                    ties[c].append(float(tie_count))
                tie_count = 1
            else:
                # Current value is tied with previous - accumulate rank
                rank_sum += float(i)
                tie_count += 1

            i += 1

        # Process the last element(s)
        for j in range(tie_count):
            idx = v_sort[i - 1 - j][1]
            x_ranked[idx, c] = (rank_sum / tie_count) + 1.0

        # Record ties for the last group if needed
        if tie_count > 1:
            ties[c].append(float(tie_count))

    return x_ranked, ties


def rank_matrix_csc(
    data: np.ndarray, indptr: np.ndarray, nrows: int, ncols: int
) -> Tuple[np.ndarray, List[List[float]]]:
    """Rank matrix in CSC format column-wise.

    Args:
        data: Values in the sparse matrix (will be modified in-place)
        indptr: Column pointers
        nrows: Number of rows
        ncols: Number of columns

    Returns:
        Tuple of (ranked_data, ties) where ties is a list of tie counts per column
    """
    ties = [[] for _ in range(ncols)]
    data_ranked = data.copy()

    for c in range(ncols):
        if indptr[c + 1] == indptr[c]:
            continue

        n_zero = nrows - (indptr[c + 1] - indptr[c])
        ties[c] = in_place_rank_mean(data_ranked, indptr[c], indptr[c + 1] - 1)
        ties[c].append(float(n_zero))

        # Add n_zero to all ranks in this column
        for j in range(indptr[c], indptr[c + 1]):
            data_ranked[j] += n_zero

    return data_ranked, ties


def rank_matrix_csr(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    nrows: int,
    ncols: int,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Rank matrix in CSR format column-wise.

    For CSR format, we need to rank by column to match CSC behavior.
    This requires creating a column-wise view of the data.

    Args:
        data: Values in the sparse matrix
        indptr: Row pointers
        indices: Column indices
        nrows: Number of rows
        ncols: Number of columns

    Returns:
        Tuple of (ranked_data, ties) where ties is a list of tie counts per column
    """
    # Create a column-wise view of the data
    col_data = [[] for _ in range(ncols)]

    # Collect values by column along with their position in data array
    for r in range(nrows):
        for j in range(indptr[r], indptr[r + 1]):
            col_idx = indices[j]
            col_data[col_idx].append((data[j], j))  # (value, position in data)

    # Now rank each column
    ties = [[] for _ in range(ncols)]
    data_ranked = data.copy()

    for c in range(ncols):
        if not col_data[c]:
            continue

        # Sort values in this column
        col_data[c].sort(key=lambda x: x[0])

        rank_sum = 0.0
        tie_count = 1.0
        i = 1

        while i < len(col_data[c]):
            if col_data[c][i][0] != col_data[c][i - 1][0]:
                # If current val != prev val, set prev val
                for j in range(int(tie_count)):
                    data_pos = col_data[c][i - 1 - j][1]
                    data_ranked[data_pos] = (rank_sum / tie_count) + 1.0

                # Restart count ranks
                rank_sum = float(i)
                if tie_count > 1.0:
                    ties[c].append(tie_count)
                tie_count = 1.0
            else:
                # If current val is a tie, compute mean rank
                rank_sum += float(i)
                tie_count += 1.0

            i += 1

        # Set the last element(s)
        for j in range(int(tie_count)):
            data_pos = col_data[c][i - 1 - j][1]
            data_ranked[data_pos] = (rank_sum / tie_count) + 1.0

        if tie_count > 1.0:
            ties[c].append(tie_count)

        # Add n_zero to all ranks in this column
        n_zero = nrows - len(col_data[c])
        ties[c].append(float(n_zero))

        for j in range(len(col_data[c])):
            data_pos = col_data[c][j][1]
            data_ranked[data_pos] += n_zero

    return data_ranked, ties
