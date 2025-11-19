"""Group-wise aggregation functions for matrices."""

import numpy as np
from scipy import sparse
from typing import Tuple


def sum_groups_dense(x: np.ndarray, groups: np.ndarray, ngroups: int) -> np.ndarray:
    """Sum values by groups for dense matrix.

    Args:
        x: Matrix where rows are samples and columns are features (nsamples × nfeatures)
        groups: Group assignments for rows (nsamples,)
        ngroups: Number of groups

    Returns:
        Summed values where rows are groups and columns are features (ngroups × nfeatures)
    """
    nfeatures = x.shape[1]
    result = np.zeros((ngroups, nfeatures), dtype=np.float64)

    # Use numpy's add.at for efficient accumulation
    np.add.at(result, groups, x)

    return result


def sum_groups_csc(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
) -> np.ndarray:
    """Sum values by groups for sparse matrix (CSC format).

    Args:
        data: Values in the sparse matrix
        indices: Row indices (i in CSC format)
        indptr: Column pointers (p in CSC format)
        ncol: Number of columns
        groups: Group assignments for rows
        ngroups: Number of groups

    Returns:
        Summed values where rows are groups and columns are features (ngroups × ncol)
    """
    result = np.zeros((ngroups, ncol), dtype=np.float64)

    for c in range(ncol):
        for j in range(indptr[c], indptr[c + 1]):
            row_idx = indices[j]
            group = groups[row_idx]
            result[group, c] += data[j]

    return result


def sum_groups_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
) -> np.ndarray:
    """Sum values by groups for sparse matrix (CSR format).

    Args:
        data: Values in the sparse matrix
        indices: Column indices (i in CSR format)
        indptr: Row pointers (p in CSR format)
        nrow: Number of rows
        ncol: Number of columns
        groups: Group assignments for rows
        ngroups: Number of groups

    Returns:
        Summed values where rows are groups and columns are features (ngroups × ncol)
    """
    result = np.zeros((ngroups, ncol), dtype=np.float64)

    for r in range(nrow):
        group = groups[r]
        for j in range(indptr[r], indptr[r + 1]):
            col_idx = indices[j]
            result[group, col_idx] += data[j]

    return result


def nnzero_groups_dense(x: np.ndarray, groups: np.ndarray, ngroups: int) -> np.ndarray:
    """Count non-zero values by groups for dense matrix.

    Args:
        x: Matrix where rows are samples and columns are features
        groups: Group assignments for rows
        ngroups: Number of groups

    Returns:
        Counted values where rows are groups and columns are features
    """
    nfeatures = x.shape[1]
    result = np.zeros((ngroups, nfeatures), dtype=np.float64)

    for c in range(nfeatures):
        for r in range(x.shape[0]):
            if x[r, c] != 0.0:
                result[groups[r], c] += 1.0

    return result


def nnzero_groups_csc(
    indptr: np.ndarray, indices: np.ndarray, ncol: int, groups: np.ndarray, ngroups: int
) -> np.ndarray:
    """Count non-zero values by groups for sparse matrix (CSC format).

    Args:
        indptr: Column pointers
        indices: Row indices
        ncol: Number of columns
        groups: Group assignments for rows
        ngroups: Number of groups

    Returns:
        Counted values where rows are groups and columns are features
    """
    result = np.zeros((ngroups, ncol), dtype=np.float64)

    for c in range(ncol):
        for j in range(indptr[c], indptr[c + 1]):
            row_idx = indices[j]
            result[groups[row_idx], c] += 1.0

    return result


def nnzero_groups_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    nrow: int,
    ncol: int,
    groups: np.ndarray,
    ngroups: int,
) -> np.ndarray:
    """Count non-zero values by groups for sparse matrix (CSR format).

    Args:
        indptr: Row pointers
        indices: Column indices
        nrow: Number of rows
        ncol: Number of columns
        groups: Group assignments for rows
        ngroups: Number of groups

    Returns:
        Counted values where rows are groups and columns are features
    """
    result = np.zeros((ngroups, ncol), dtype=np.float64)

    for r in range(nrow):
        group = groups[r]
        for j in range(indptr[r], indptr[r + 1]):
            col_idx = indices[j]
            result[group, col_idx] += 1.0

    return result
