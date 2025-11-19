"""Main Wilcoxon rank-sum test implementation."""

import numpy as np
from scipy import sparse
from typing import List, Optional, Union
from collections import Counter

from .models import WilcoxResult
from .matrix import Matrix, MatrixFormat
from .ranking import rank_matrix_dense, rank_matrix_csc, rank_matrix_csr
from .stats import compute_pval, compute_fdr


def _set_comparison_label(group: str, comparison: Optional[str]) -> str:
    """Set comparison string pattern "group1/group2".

    Args:
        group: Group1 label
        comparison: Comparison string without knowing the group1 and group2 order

    Returns:
        Correct comparison label
    """
    if comparison:
        if "_|_" in comparison:
            pairs = comparison.split("_|_")
            if group == pairs[0]:
                return f"{pairs[0]}_vs_{pairs[1]}"
            return f"{pairs[1]}_vs_{pairs[0]}"
        return f"{group}_vs_rest"
    else:
        return f"{group}_vs_rest"


def _compute_ustat_dense(
    x_ranked: np.ndarray, groups: np.ndarray, group_size: np.ndarray
) -> np.ndarray:
    """Compute U statistic for dense matrix.

    Args:
        x_ranked: Ranked matrix (nsamples × nfeatures)
        groups: Group indices (nsamples,)
        group_size: Size of each group (ngroups,)

    Returns:
        U statistic matrix (ngroups × nfeatures)
    """
    ngroups = len(group_size)
    nfeatures = x_ranked.shape[1]

    # Sum ranks by group
    grs = np.zeros((ngroups, nfeatures), dtype=np.float64)
    np.add.at(grs, groups, x_ranked)

    # Calculate U statistic
    # ustat = grs - group_size * (group_size + 1) / 2
    ustat = grs - group_size[:, np.newaxis] * (group_size[:, np.newaxis] + 1) / 2

    return ustat


def _compute_ustat_csc(
    x_ranked: sparse.csc_matrix, groups: np.ndarray, group_size: np.ndarray
) -> np.ndarray:
    """Compute U statistic for CSC matrix.

    Args:
        x_ranked: Ranked CSC matrix
        groups: Group indices
        group_size: Size of each group

    Returns:
        U statistic matrix (ngroups × nfeatures)
    """
    ngroups = len(group_size)
    ncols = x_ranked.shape[1]

    # Calculate group rank sums
    grs = np.zeros((ngroups, ncols), dtype=np.float64)
    for c in range(ncols):
        for j in range(x_ranked.indptr[c], x_ranked.indptr[c + 1]):
            row_idx = x_ranked.indices[j]
            group = groups[row_idx]
            grs[group, c] += x_ranked.data[j]

    # Calculate non-zero counts per group
    gnz = np.zeros((ngroups, ncols), dtype=np.float64)
    for c in range(ncols):
        for j in range(x_ranked.indptr[c], x_ranked.indptr[c + 1]):
            row_idx = x_ranked.indices[j]
            gnz[groups[row_idx], c] += 1.0

    # Get number of zeros per group
    group_sizes_2d = np.tile(group_size[:, np.newaxis], (1, ncols))
    gnz = group_sizes_2d - gnz

    # Calculate average rank for zeros in each column
    zero_ranks = np.array(
        [(x_ranked.shape[0] - (x_ranked.indptr[c + 1] - x_ranked.indptr[c]) + 1) / 2.0
         for c in range(ncols)]
    )

    # Calculate U statistic
    ustat = np.zeros((ngroups, ncols), dtype=np.float64)
    for g in range(ngroups):
        gs = group_size[g]
        ustat[g, :] = gnz[g, :] * zero_ranks + grs[g, :] - gs * (gs + 1.0) / 2.0

    return ustat


def _compute_ustat_csr(
    x_ranked: sparse.csr_matrix, groups: np.ndarray, group_size: np.ndarray
) -> np.ndarray:
    """Compute U statistic for CSR matrix.

    Args:
        x_ranked: Ranked CSR matrix
        groups: Group indices
        group_size: Size of each group

    Returns:
        U statistic matrix (ngroups × nfeatures)
    """
    ngroups = len(group_size)
    nrows, ncols = x_ranked.shape

    # Calculate group rank sums
    grs = np.zeros((ngroups, ncols), dtype=np.float64)
    for r in range(nrows):
        group = groups[r]
        for j in range(x_ranked.indptr[r], x_ranked.indptr[r + 1]):
            col_idx = x_ranked.indices[j]
            grs[group, col_idx] += x_ranked.data[j]

    # Calculate non-zero counts per group
    gnz = np.zeros((ngroups, ncols), dtype=np.float64)
    for r in range(nrows):
        group = groups[r]
        for j in range(x_ranked.indptr[r], x_ranked.indptr[r + 1]):
            col_idx = x_ranked.indices[j]
            gnz[group, col_idx] += 1.0

    # Get number of zeros per group
    group_sizes_2d = np.tile(group_size[:, np.newaxis], (1, ncols))
    gnz = group_sizes_2d - gnz

    # Calculate non-zeros per column
    nnz_per_col = np.zeros(ncols, dtype=np.int64)
    for r in range(nrows):
        for j in range(x_ranked.indptr[r], x_ranked.indptr[r + 1]):
            nnz_per_col[x_ranked.indices[j]] += 1

    # Calculate average rank for zeros in each column
    zero_ranks = (nrows - nnz_per_col + 1) / 2.0

    # Calculate U statistic
    ustat = np.zeros((ngroups, ncols), dtype=np.float64)
    for g in range(ngroups):
        gs = group_size[g]
        ustat[g, :] = gnz[g, :] * zero_ranks + grs[g, :] - gs * (gs + 1.0) / 2.0

    return ustat


def wilcoxauc(
    x: Union[np.ndarray, sparse.spmatrix],
    y: np.ndarray,
    groups_use: Optional[List[str]] = None,
    y_order: Optional[List[str]] = None,
    comparison: Optional[str] = None,
) -> List[WilcoxResult]:
    """Perform Wilcoxon rank-sum test and calculate auROC.

    Args:
        x: Matrix where rows are samples and columns are features
        y: Vector of group labels (strings)
        groups_use: Optional subset of groups in y to test
        y_order: Optional order of groups in y
        comparison: Optional comparison pattern

    Returns:
        List of WilcoxResult objects containing test results
    """
    # Convert inputs to numpy arrays
    y = np.asarray(y)
    n_samples = x.shape[0]

    assert n_samples == len(y), "Number of rows in X must match length of y"

    # Filter by groups_use if provided
    if groups_use is not None:
        mask = np.isin(y, groups_use)
        if sparse.issparse(x):
            x = x[mask, :]
        else:
            x = x[mask, :]
        y = y[mask]

    # Remove empty string labels (treated as NA)
    mask = y != ""
    if not mask.all():
        if sparse.issparse(x):
            x = x[mask, :]
        else:
            x = x[mask, :]
        y = y[mask]

    # Get unique groups and counts
    group_counts = Counter(y)
    unique_groups = list(group_counts.keys())

    # Apply y_order if provided
    if y_order is not None:
        if groups_use is not None:
            y_order_filtered = [g for g in y_order if g in groups_use]
        else:
            y_order_filtered = [g for g in y_order if g in unique_groups]
        group_idxs = y_order_filtered
    else:
        group_idxs = unique_groups

    # Check if we have at least 2 groups
    if len(group_idxs) < 2:
        raise ValueError("Must have at least 2 groups defined")

    # Create group mapping
    group_map = {group: idx for idx, group in enumerate(group_idxs)}

    # Convert group labels to indices
    group_indices = np.array([group_map[label] for label in y])

    # Get group sizes
    group_size = np.array([group_counts[g] for g in group_idxs], dtype=np.float64)

    # Calculate n1n2 (product of group size and complement size)
    n1n2 = group_size * (len(y) - group_size)

    # Wrap matrix
    x_matrix = Matrix.from_anndata(x)

    # Rank the matrix
    if x_matrix.format == MatrixFormat.DENSE:
        x_ranked, ties = rank_matrix_dense(x_matrix.data)
    elif x_matrix.format == MatrixFormat.CSC:
        data_ranked, ties = rank_matrix_csc(
            x_matrix.data.data.copy(),
            x_matrix.data.indptr,
            x_matrix.data.shape[0],
            x_matrix.data.shape[1],
        )
        x_ranked = sparse.csc_matrix(
            (data_ranked, x_matrix.data.indices, x_matrix.data.indptr),
            shape=x_matrix.data.shape,
        )
    else:  # CSR
        data_ranked, ties = rank_matrix_csr(
            x_matrix.data.data.copy(),
            x_matrix.data.indptr,
            x_matrix.data.indices,
            x_matrix.data.shape[0],
            x_matrix.data.shape[1],
        )
        x_ranked = sparse.csr_matrix(
            (data_ranked, x_matrix.data.indices, x_matrix.data.indptr),
            shape=x_matrix.data.shape,
        )

    # Compute U statistic
    if x_matrix.format == MatrixFormat.DENSE:
        ustat = _compute_ustat_dense(x_ranked, group_indices, group_size)
    elif x_matrix.format == MatrixFormat.CSC:
        ustat = _compute_ustat_csc(x_ranked, group_indices, group_size)
    else:  # CSR
        ustat = _compute_ustat_csr(x_ranked, group_indices, group_size)

    # Calculate p-values
    pvals = compute_pval(ustat, ties, len(y), n1n2)

    # Calculate FDR (Benjamini-Hochberg adjustment)
    fdr = compute_fdr(pvals)

    # Calculate AUC
    auc = ustat / n1n2[:, np.newaxis]

    # Calculate auxiliary statistics
    # Group sums
    group_sums = x_matrix.sum_groups(group_indices, len(group_idxs))

    # Non-zero counts per group
    group_nnz = x_matrix.nnzero_groups(group_indices, len(group_idxs))

    # Calculate percentages
    group_pct = (group_nnz / group_size[:, np.newaxis]) * 100.0

    # Calculate out-of-group percentages
    total_nnz = group_nnz.sum(axis=0)
    group_pct_out = (
        (total_nnz - group_nnz) / (len(y) - group_size[:, np.newaxis])
    ) * 100.0

    # Calculate group means
    group_means = group_sums / group_size[:, np.newaxis]

    # Calculate log fold change
    total_sum = group_sums.sum(axis=0)
    lfc = np.zeros_like(group_means)
    for g in range(len(group_idxs)):
        out_sum = total_sum - group_sums[g, :]
        out_mean = out_sum / (len(y) - group_size[g])
        lfc[g, :] = group_means[g, :] - out_mean

    # Assemble results
    results = []
    for r in range(ustat.shape[0]):
        for c in range(ustat.shape[1]):
            results.append(
                WilcoxResult(
                    feature=c,
                    group=group_idxs[r],
                    avg_expr=group_means[r, c],
                    log_fc=lfc[r, c],
                    ustat=ustat[r, c],
                    auc=auc[r, c],
                    pval=pvals[r, c],
                    padj=fdr[r, c],
                    pct_in=group_pct[r, c],
                    pct_out=group_pct_out[r, c],
                    comparison=_set_comparison_label(group_idxs[r], comparison),
                )
            )

    return results
