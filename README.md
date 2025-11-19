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

## Release

Publishing to PyPI is automated via GitHub Actions defined in `.github/workflows/ci.yml`.

1. Create a PyPI API token with scope "Project: fastwilcox" at <https://pypi.org/manage/account/token/>.
2. In the repository, navigate to *Settings → Secrets and variables → Actions* and add a secret named `PYPI_API_TOKEN` containing the token value.
3. Bump the version in both `pyproject.toml` and `fastwilcox/__init__.py`, commit, and push to `main`.
4. Publish a Git tag and GitHub release (e.g., `v0.1.1`), or trigger the workflow manually from the Actions tab via **CI and Publish → Run workflow**.
5. The workflow runs the full test matrix, builds the sdist/wheel artifacts, and uploads them to PyPI when the release job succeeds.

To target TestPyPI instead, set the `repository-url` input of the publish step to `https://test.pypi.org/legacy/` and provide a `TEST_PYPI_API_TOKEN` secret.

## Performance

This implementation is optimized for:

- Large sparse matrices (common in single-cell RNA-seq)
- Efficient group-wise operations
- Vectorized numpy operations

For the best performance with sparse data, use CSC format.

## License

MIT License

## Citation

If you use this software, please cite the Toucan project.
