"""Matrix wrapper supporting Dense, CSR, and CSC formats."""

import numpy as np
from scipy import sparse
from typing import Union, Tuple
from enum import Enum

from .grouping import (
    sum_groups_dense,
    sum_groups_csc,
    sum_groups_csr,
    nnzero_groups_dense,
    nnzero_groups_csc,
    nnzero_groups_csr,
)


class MatrixFormat(Enum):
    """Matrix format types."""

    DENSE = "dense"
    CSR = "csr"
    CSC = "csc"


class Matrix:
    """Wrapper class for matrices supporting Dense, CSR, and CSC formats."""

    def __init__(self, data: Union[np.ndarray, sparse.csr_matrix, sparse.csc_matrix]):
        """Initialize matrix wrapper.

        Args:
            data: Matrix data (numpy array, CSR matrix, or CSC matrix)
        """
        if isinstance(data, np.ndarray):
            self.format = MatrixFormat.DENSE
            self.data = data
        elif sparse.isspmatrix_csr(data):
            self.format = MatrixFormat.CSR
            self.data = data
        elif sparse.isspmatrix_csc(data):
            self.format = MatrixFormat.CSC
            self.data = data
        else:
            raise TypeError(f"Unsupported matrix type: {type(data)}")

    @classmethod
    def from_anndata(cls, array_data) -> "Matrix":
        """Create Matrix from AnnData ArrayData.

        Args:
            array_data: AnnData array (can be dense or sparse)

        Returns:
            Matrix wrapper
        """
        if sparse.issparse(array_data):
            if sparse.isspmatrix_csc(array_data):
                return cls(array_data.astype(np.float64))
            elif sparse.isspmatrix_csr(array_data):
                return cls(array_data.astype(np.float64))
            else:
                # Convert to CSC by default
                return cls(array_data.tocsc().astype(np.float64))
        else:
            return cls(np.asarray(array_data, dtype=np.float64))

    def sum_groups(self, groups: np.ndarray, ngroups: int) -> np.ndarray:
        """Sum values by groups.

        Args:
            groups: Group assignments for rows
            ngroups: Number of groups

        Returns:
            Summed values (ngroups × nfeatures)
        """
        if self.format == MatrixFormat.DENSE:
            return sum_groups_dense(self.data, groups, ngroups)
        elif self.format == MatrixFormat.CSC:
            return sum_groups_csc(
                self.data.data,
                self.data.indices,
                self.data.indptr,
                self.data.shape[1],
                groups,
                ngroups,
            )
        else:  # CSR
            return sum_groups_csr(
                self.data.data,
                self.data.indices,
                self.data.indptr,
                self.data.shape[0],
                self.data.shape[1],
                groups,
                ngroups,
            )

    def nnzero_groups(self, groups: np.ndarray, ngroups: int) -> np.ndarray:
        """Count non-zero values by groups.

        Args:
            groups: Group assignments for rows
            ngroups: Number of groups

        Returns:
            Non-zero counts (ngroups × nfeatures)
        """
        if self.format == MatrixFormat.DENSE:
            return nnzero_groups_dense(self.data, groups, ngroups)
        elif self.format == MatrixFormat.CSC:
            return nnzero_groups_csc(
                self.data.indptr, self.data.indices, self.data.shape[1], groups, ngroups
            )
        else:  # CSR
            return nnzero_groups_csr(
                self.data.indptr,
                self.data.indices,
                self.data.shape[0],
                self.data.shape[1],
                groups,
                ngroups,
            )

    def nrows(self) -> int:
        """Get number of rows."""
        return self.data.shape[0]

    def ncols(self) -> int:
        """Get number of columns."""
        return self.data.shape[1]

    def nnz_per_col(self) -> np.ndarray:
        """Get number of non-zeros per column.

        Returns:
            Array of non-zero counts per column
        """
        if self.format == MatrixFormat.DENSE:
            # For dense matrix, this is just the number of rows
            return np.full(self.ncols(), self.nrows(), dtype=np.float64)
        elif self.format == MatrixFormat.CSC:
            # For CSC, this is easy - just the difference in indptr
            return np.diff(self.data.indptr).astype(np.float64)
        else:  # CSR
            # For CSR, we need to count per column
            nnz = np.zeros(self.ncols(), dtype=np.float64)
            for row_idx in range(self.nrows()):
                start = self.data.indptr[row_idx]
                end = self.data.indptr[row_idx + 1]
                for j in range(start, end):
                    col_idx = self.data.indices[j]
                    nnz[col_idx] += 1.0
            return nnz
