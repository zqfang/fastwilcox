"""Tests for Wilcoxon test with dense matrices."""

import pytest
import numpy as np
from fastwilcox import wilcoxauc


def test_wilcoxauc_dense_basic(dense_h5ad, dense_reference, compare_results):
    """Test wilcoxauc with dense matrix against reference results."""
    # Get data
    x = dense_h5ad.X
    y = dense_h5ad.obs["cell.type"].values

    # Get unique categories in order
    if hasattr(dense_h5ad.obs["cell.type"], "cat"):
        y_order = dense_h5ad.obs["cell.type"].cat.categories.tolist()
    else:
        y_order = None

    # Run wilcoxauc
    results = wilcoxauc(x, y, groups_use=None, y_order=y_order, comparison=None)

    # Get variable names
    var_names = dense_h5ad.var_names.tolist()

    # Compare with reference
    assert compare_results(
        results, dense_reference, var_names, tolerance=1e-5
    ), "Dense results do not match reference"


def test_wilcoxauc_dense_simple():
    """Test wilcoxauc with a simple example."""
    # Create simple test data
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array(["A", "B", "A"])

    # Run wilcoxauc
    results = wilcoxauc(x, y)

    # Basic checks
    assert len(results) > 0, "Should return results"
    assert all(r.feature in [0, 1, 2] for r in results), "Feature indices should be valid"
    assert all(r.group in ["A", "B"] for r in results), "Groups should be A or B"

    # Check that we have results for both groups and all features
    groups = {r.group for r in results}
    assert groups == {"A", "B"}, "Should have results for both groups"

    features = {r.feature for r in results}
    assert features == {0, 1, 2}, "Should have results for all features"


def test_wilcoxauc_dense_groups_filter(dense_h5ad):
    """Test wilcoxauc with groups_use filter."""
    # Get data
    x = dense_h5ad.X
    y = dense_h5ad.obs["cell.type"].values

    # Get unique groups
    unique_groups = np.unique(y)
    if len(unique_groups) > 2:
        # Test with subset of groups
        groups_use = unique_groups[:2].tolist()

        results = wilcoxauc(x, y, groups_use=groups_use)

        # Check that only selected groups appear in results
        result_groups = {r.group for r in results}
        assert result_groups.issubset(
            set(groups_use)
        ), "Results should only contain selected groups"


def test_wilcoxauc_dense_invalid_input():
    """Test wilcoxauc with invalid inputs."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Mismatched dimensions
    y = np.array(["A", "B", "C"])
    with pytest.raises(AssertionError):
        wilcoxauc(x, y)

    # Only one group
    y = np.array(["A", "A"])
    with pytest.raises(ValueError, match="Must have at least 2 groups"):
        wilcoxauc(x, y)


def test_wilcoxauc_dense_pvalues(dense_h5ad):
    """Test that p-values are in valid range."""
    x = dense_h5ad.X
    y = dense_h5ad.obs["cell.type"].values

    results = wilcoxauc(x, y)

    # Check p-values are in [0, 1]
    for r in results:
        assert 0 <= r.pval <= 1, f"P-value {r.pval} out of range"
        assert 0 <= r.padj <= 1, f"Adjusted p-value {r.padj} out of range"


def test_wilcoxauc_dense_auc(dense_h5ad):
    """Test that AUC values are in valid range."""
    x = dense_h5ad.X
    y = dense_h5ad.obs["cell.type"].values

    results = wilcoxauc(x, y)

    # Check AUC values are in [0, 1]
    for r in results:
        assert 0 <= r.auc <= 1, f"AUC {r.auc} out of range"


def test_wilcoxauc_dense_percentages(dense_h5ad):
    """Test that percentage values are in valid range."""
    x = dense_h5ad.X
    y = dense_h5ad.obs["cell.type"].values

    results = wilcoxauc(x, y)

    # Check percentages are in [0, 100]
    for r in results:
        assert 0 <= r.pct_in <= 100, f"Pct_in {r.pct_in} out of range"
        assert 0 <= r.pct_out <= 100, f"Pct_out {r.pct_out} out of range"
