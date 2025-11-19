# FastWilcox

Fast Wilcoxon rank-sum test for differential expression analysis in single-cell genomics.

This is a Python implementation of the `wilcoxauc` from the R' [presto package](https://github.com/immunogenomics/presto), providing high-performance statistical testing for single-cell RNA-seq data.

## Features

- **Fast Wilcoxon rank-sum test** with AUC calculation
- **Multiple matrix formats** supported: Dense, CSR, and CSC sparse matrices
- **Optimized for single-cell data** with efficient handling of sparse matrices
- **FDR correction** using Benjamini-Hochberg method
- **Compatible with AnnData** objects

## Installation

```bash
cd python
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import anndata as ad
from fastwilcox import wilcoxauc

# Load your data
adata = ad.read_h5ad("your_data.h5ad")

# Run Wilcoxon test
results = wilcoxauc(
    x=adata.X,
    y=adata.obs['cell_type'].values,
    groups_use=None,  # Optional: test specific groups
    y_order=None,     # Optional: specify group order
    comparison=None   # Optional: pairwise comparison label
)

# Results is a list of WilcoxResult objects
for res in results:
    print(f"Feature {res.feature}, Group {res.group}: "
          f"logFC={res.log_fc:.3f}, AUC={res.auc:.3f}, "
          f"p-adj={res.padj:.3e}")
```

## API Reference

### `wilcoxauc(x, y, groups_use=None, y_order=None, comparison=None)`

Perform Wilcoxon rank-sum test for differential expression.

**Parameters:**
- `x`: Matrix (Dense array, CSR, or CSC sparse matrix) where rows are samples and columns are features
- `y`: List/array of group labels (strings)
- `groups_use`: Optional list of groups to test (default: all groups)
- `y_order`: Optional list specifying the order of groups
- `comparison`: Optional comparison pattern for pairwise tests

**Returns:**
- List of `WilcoxResult` objects with fields:
  - `feature`: Feature index
  - `group`: Group name
  - `avg_expr`: Average expression in group
  - `log_fc`: Log fold change (in-group mean - out-group mean)
  - `ustat`: U-statistic
  - `auc`: Area under ROC curve
  - `pval`: P-value
  - `padj`: FDR-adjusted p-value
  - `pct_in`: Percentage of cells expressing in group
  - `pct_out`: Percentage of cells expressing out of group
  - `comparison`: Comparison label, useful when only compare two group

## Testing

Run tests:

```bash
cd python
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=fastwilcox --cov-report=html
```

## Performance

This implementation is optimized for:

- Large sparse matrices (common in single-cell RNA-seq)
- Efficient group-wise operations
- Vectorized numpy operations

For the best performance with sparse data, use CSC format.

## License

MIT License

## Citation

If you use this software, please cite the fastwilcox project.
