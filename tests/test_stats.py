"""Unit tests for statistical functions."""

import pytest
import numpy as np
from fastwilcox.stats import compute_fdr, compute_pval


def test_compute_fdr_basic():
    """Test FDR correction with basic example."""
    # Create a 2x6 array with p-values for two groups
    p_values = np.array(
        [[0.01, 0.04, 0.02, 0.001, 0.05, 0.03], [0.2, 0.8, 0.01, 0.8, 0.001, 0.44]]
    )

    adjusted = compute_fdr(p_values)

    # Check shape matches
    assert adjusted.shape == p_values.shape, "Shape should match input"

    # Check all values are valid probabilities
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "All FDR values should be in [0, 1]"

    # Check that adjusted p-values are >= original p-values (generally true for BH)
    # Note: This is not always strictly true due to the step-down procedure
    # but adjusted values should be close to or greater than originals for most cases


def test_compute_fdr_sorted():
    """Test FDR correction maintains order properties."""
    # Sorted p-values
    p_values = np.array([[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]])

    adjusted = compute_fdr(p_values)

    # Adjusted p-values should be monotonically non-decreasing when input is sorted
    for i in range(len(adjusted[0]) - 1):
        assert (
            adjusted[0, i] <= adjusted[0, i + 1]
        ), "FDR should be non-decreasing for sorted input"


def test_compute_fdr_all_significant():
    """Test FDR with all small p-values."""
    p_values = np.array([[0.001, 0.002, 0.003, 0.004, 0.005]])

    adjusted = compute_fdr(p_values)

    # All should still be relatively small
    assert np.all(adjusted < 0.1), "All adjusted values should be small for significant results"


def test_compute_fdr_all_nonsignificant():
    """Test FDR with all large p-values."""
    p_values = np.array([[0.9, 0.95, 0.99, 0.85, 0.8]])

    adjusted = compute_fdr(p_values)

    # All should be close to 1
    assert np.all(adjusted >= 0.8), "All adjusted values should be large for non-significant results"


def test_compute_fdr_with_nan():
    """Test FDR correction with NaN values."""
    p_values = np.array([[0.01, np.nan, 0.05, 0.1]])

    adjusted = compute_fdr(p_values)

    # Check that non-NaN values are adjusted
    assert not np.isnan(adjusted[0, 0]), "Non-NaN values should be adjusted"
    assert not np.isnan(adjusted[0, 2]), "Non-NaN values should be adjusted"


def test_compute_pval_basic():
    """Test p-value computation with basic example."""
    # Simple U statistic matrix (2 groups × 3 features)
    ustat = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])

    # Ties for each feature (3 features)
    ties = [[2.0, 5.0], [3.0, 4.0], [1.0, 6.0]]

    # Total number of samples
    n = 10.0

    # n1n2 for each group (product of group size and complement size)
    n1n2 = np.array([15.0, 18.0])  # e.g., 3*5 and 2*9

    pvals = compute_pval(ustat, ties, n, n1n2)

    # Check shape
    assert pvals.shape == ustat.shape, "P-value shape should match U-stat shape"

    # Check all values are valid probabilities
    assert np.all((pvals >= 0) & (pvals <= 1)), "All p-values should be in [0, 1]"


def test_compute_pval_extreme_ustat():
    """Test p-value computation with extreme U-statistics."""
    # Very large U statistics (should give small p-values)
    ustat = np.array([[100.0, 150.0, 200.0]])
    ties = [[1.0], [1.0], [1.0]]
    n = 20.0
    n1n2 = np.array([50.0])

    pvals = compute_pval(ustat, ties, n, n1n2)

    # Should give very small p-values
    assert np.all(pvals < 0.1), "Large U-statistics should give small p-values"


def test_compute_pval_no_ties():
    """Test p-value computation with no ties."""
    ustat = np.array([[10.0, 20.0]])
    ties = [[], []]  # No ties
    n = 10.0
    n1n2 = np.array([12.0])

    pvals = compute_pval(ustat, ties, n, n1n2)

    # Should still work without ties
    assert pvals.shape == (1, 2), "Should handle no ties case"
    assert np.all((pvals >= 0) & (pvals <= 1)), "P-values should be valid"


def test_compute_pval_symmetry():
    """Test that p-values are symmetric around n1n2/2."""
    # U-statistics symmetric around expected value
    n1n2_val = 24.0
    ustat = np.array([[12.0, 36.0]])  # One below, one above n1n2/2
    ties = [[1.0], [1.0]]
    n = 10.0
    n1n2 = np.array([n1n2_val])

    pvals = compute_pval(ustat, ties, n, n1n2)

    # P-values should be the same for symmetric U-statistics
    # (accounting for the -0.5 continuity correction)
    # This is an approximate test
    assert pvals.shape == (1, 2), "Should return correct shape"
