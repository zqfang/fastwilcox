"""Tests for Wilcoxon test with CSC sparse matrices."""

import pytest
import numpy as np
from scipy import sparse
from fastwilcox import wilcoxauc


def test_wilcoxauc_csc_basic(csc_h5ad, csc_reference, compare_results):
    """Test wilcoxauc with CSC sparse matrix against reference results."""
    # Get data
    x = csc_h5ad.X
    y = csc_h5ad.obs["cell.type"].values

    # Verify it's CSC format
    assert sparse.isspmatrix_csc(x), "Data should be in CSC format"

    # Get unique categories in order
    if hasattr(csc_h5ad.obs["cell.type"], "cat"):
        y_order = csc_h5ad.obs["cell.type"].cat.categories.tolist()
    else:
        y_order = None

    # Run wilcoxauc
    results = wilcoxauc(x, y, groups_use=None, y_order=y_order, comparison=None)

    # Get variable names
    var_names = csc_h5ad.var_names.tolist()

    # Compare with reference
    assert compare_results(
        results, csc_reference, var_names, tolerance=1e-5
    ), "CSC results do not match reference"


def test_wilcoxauc_csc_simple():
    """Test wilcoxauc with a simple CSC sparse matrix."""
    # Create simple sparse test data
    x = sparse.csc_matrix(
        np.array([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 8.0, 0.0]])
    )
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


def test_wilcoxauc_csc_all_zeros():
    """Test wilcoxauc with columns that are all zeros."""
    # Create sparse matrix with a zero column
    x = sparse.csc_matrix(
        np.array([[1.0, 0.0, 3.0], [4.0, 0.0, 6.0], [7.0, 0.0, 9.0]])
    )
    y = np.array(["A", "B", "A"])

    # Run wilcoxauc
    results = wilcoxauc(x, y)

    # Should still have results for all features
    features = {r.feature for r in results}
    assert features == {0, 1, 2}, "Should have results for all features including zero column"


def test_wilcoxauc_csc_vs_dense(csc_h5ad):
    """Test that CSC and dense versions give same results."""
    # Get CSC data
    x_csc = csc_h5ad.X
    y = csc_h5ad.obs["cell.type"].values

    # Get unique categories in order
    if hasattr(csc_h5ad.obs["cell.type"], "cat"):
        y_order = csc_h5ad.obs["cell.type"].cat.categories.tolist()
    else:
        y_order = None

    # Convert to dense
    x_dense = x_csc.toarray()

    # Run both versions
    results_csc = wilcoxauc(x_csc, y, y_order=y_order)
    results_dense = wilcoxauc(x_dense, y, y_order=y_order)

    # Compare results
    assert len(results_csc) == len(
        results_dense
    ), "CSC and dense should return same number of results"

    for r_csc, r_dense in zip(results_csc, results_dense):
        assert r_csc.feature == r_dense.feature, "Feature mismatch"
        assert r_csc.group == r_dense.group, "Group mismatch"
        assert np.isclose(
            r_csc.ustat, r_dense.ustat, rtol=1e-5
        ), f"U-stat mismatch: {r_csc.ustat} vs {r_dense.ustat}"
        assert np.isclose(
            r_csc.auc, r_dense.auc, rtol=1e-5
        ), f"AUC mismatch: {r_csc.auc} vs {r_dense.auc}"
        assert np.isclose(
            r_csc.pval, r_dense.pval, rtol=1e-5
        ), f"P-value mismatch: {r_csc.pval} vs {r_dense.pval}"


def test_wilcoxauc_csc_pvalues(csc_h5ad):
    """Test that p-values are in valid range."""
    x = csc_h5ad.X
    y = csc_h5ad.obs["cell.type"].values

    results = wilcoxauc(x, y)

    # Check p-values are in [0, 1]
    for r in results:
        assert 0 <= r.pval <= 1, f"P-value {r.pval} out of range"
        assert 0 <= r.padj <= 1, f"Adjusted p-value {r.padj} out of range"


def test_wilcoxauc_csc_auc(csc_h5ad):
    """Test that AUC values are in valid range."""
    x = csc_h5ad.X
    y = csc_h5ad.obs["cell.type"].values

    results = wilcoxauc(x, y)

    # Check AUC values are in [0, 1]
    for r in results:
        assert 0 <= r.auc <= 1, f"AUC {r.auc} out of range"
