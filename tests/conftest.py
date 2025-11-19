"""Pytest configuration and fixtures for testing."""

import pytest
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    repo_root = Path(__file__).resolve().parents[1]

    return repo_root / "tests" / "data"

@pytest.fixture
def dense_h5ad(test_data_dir):
    """Load dense test data."""
    return ad.read_h5ad(test_data_dir / "small.test.dense.h5ad")


@pytest.fixture
def csc_h5ad(test_data_dir):
    """Load CSC test data."""
    return ad.read_h5ad(test_data_dir / "small.test.h5ad")


@pytest.fixture
def csr_h5ad(test_data_dir):
    """Load CSR test data."""
    return ad.read_h5ad(test_data_dir / "small.test.csr.h5ad")


@pytest.fixture
def dense_reference(test_data_dir):
    """Load dense reference results."""
    return pd.read_csv(
        test_data_dir / "wilcox_results_dense.txt",
        sep="\t",
        dtype={"Feature": str},
    )


@pytest.fixture
def csc_reference(test_data_dir):
    """Load CSC reference results."""
    return pd.read_csv(
        test_data_dir / "wilcox_results_csc.txt",
        sep="\t",
        dtype={"Feature": str},
    )


@pytest.fixture
def csr_reference(test_data_dir):
    """Load CSR reference results."""
    return pd.read_csv(
        test_data_dir / "wilcox_results_csr.txt",
        sep="\t",
        dtype={"Feature": str},
    )


@pytest.fixture
def compare_results():
    """Return compare_results function as a fixture."""
    def _compare(py_results, ref_df, var_names, tolerance=1e-5):
        """Compare Python results with reference data.

        Args:
            py_results: List of WilcoxResult objects from Python
            ref_df: Reference DataFrame from Rust
            var_names: List of variable names
            tolerance: Tolerance for floating point comparisons

        Returns:
            bool: True if results match within tolerance
        """
        # Convert Python results to DataFrame for easier comparison
        py_df = pd.DataFrame(
            [
                {
                    "Feature": var_names[r.feature],
                    "Group": r.group,
                    "AvgExpr": r.avg_expr,
                    "LogFC": r.log_fc,
                    "UStat": r.ustat,
                    "AUC": r.auc,
                    "Pval": r.pval,
                    "Padj": r.padj,
                    "Pct_In": r.pct_in,
                    "Pct_Out": r.pct_out,
                }
                for r in py_results
            ]
        )

        # Sort both DataFrames by Feature and Group for consistent comparison
        py_df = py_df.sort_values(["Feature", "Group"]).reset_index(drop=True)
        ref_df = ref_df.sort_values(["Feature", "Group"]).reset_index(drop=True)

        # Check lengths match
        if len(py_df) != len(ref_df):
            print(f"Length mismatch: Python={len(py_df)}, Reference={len(ref_df)}")
            return False

        # Compare each numeric column
        numeric_cols = ["AvgExpr", "LogFC", "UStat", "AUC", "Pval", "Padj", "Pct_In", "Pct_Out"]

        mismatches = []
        for col in numeric_cols:
            diff = np.abs(py_df[col].values - ref_df[col].values)
            max_diff = diff.max()

            if max_diff > tolerance:
                mismatches.append((col, max_diff))
                # Find the row with maximum difference
                max_idx = diff.argmax()
                print(f"\n{col} mismatch (max diff: {max_diff:.2e}):")
                print(f"  Feature: {py_df.loc[max_idx, 'Feature']}")
                print(f"  Group: {py_df.loc[max_idx, 'Group']}")
                print(f"  Python: {py_df.loc[max_idx, col]:.6e}")
                print(f"  Reference: {ref_df.loc[max_idx, col]:.6e}")

        if mismatches:
            print(f"\nFound {len(mismatches)} mismatches exceeding tolerance {tolerance}")
            return False

        return True

    return _compare
