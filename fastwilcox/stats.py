"""Statistical functions for Wilcoxon test."""

import numpy as np
from scipy import stats
from typing import List


def compute_fdr(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction row-wise on a 2D array.

    Args:
        pvals: 2D array of p-values (ngroups × nfeatures)

    Returns:
        2D array of FDR-adjusted p-values
    """
    fdr = np.zeros_like(pvals)

    for r in range(pvals.shape[0]):
        # Extract p-values with their original indices
        p_values = [(i, p) for i, p in enumerate(pvals[r, :]) if not np.isnan(p)]

        if not p_values:
            continue  # Skip if all values are NaN

        # Sort by p-value descending
        p_values.sort(key=lambda x: x[1], reverse=True)

        # Apply BH correction
        n = len(p_values)
        min_adjusted = np.inf

        for rank, (idx, p) in enumerate(p_values):
            adjusted = (p * n) / (n - rank)
            min_adjusted = min(min_adjusted, adjusted)
            fdr[r, idx] = min(min_adjusted, 1.0)

    return fdr


def compute_pval(
    ustat: np.ndarray, ties: List[List[float]], n: float, n1n2: np.ndarray
) -> np.ndarray:
    """Compute p-values for Wilcoxon test using normal approximation.

    Args:
        ustat: U statistic matrix (ngroups × nfeatures)
        ties: Vector of ties for each feature
        n: Total number of samples
        n1n2: Product of group size and complement size for each group (ngroups,)

    Returns:
        Matrix of p-values (ngroups × nfeatures)
    """
    # z = ustat - 0.5 * n1n2
    z = ustat - 0.5 * n1n2[:, np.newaxis]

    # z = z - sign(z) * 0.5
    z = z - np.sign(z) * 0.5

    # Calculate variance components
    x1 = n**3 - n
    x2 = 1.0 / (12.0 * (n**2 - n))

    # Calculate right-hand side for each column
    rhs = np.zeros(len(ties))
    for j, tvals in enumerate(ties):
        tie_correction = sum(t**3 - t for t in tvals)
        rhs[j] = (x1 - tie_correction) * x2

    # Calculate standard deviation (usigma)
    usigma = np.sqrt(n1n2[:, np.newaxis] * rhs[np.newaxis, :])

    # z = z / usigma
    z = z / usigma

    # Calculate p-values: 2 * Φ(-|z|)
    pvals = 2.0 * stats.norm.cdf(-np.abs(z))

    return pvals
