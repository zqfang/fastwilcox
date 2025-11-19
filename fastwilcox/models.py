"""Data models for Wilcoxon test results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WilcoxResult:
    """Result structure for Wilcoxon rank-sum test.

    Attributes:
        feature: Feature index
        group: Group name
        avg_expr: Average expression in group
        log_fc: Log fold change (in-group mean - out-group mean)
        ustat: Mann-Whitney U-statistic
        auc: Area under ROC curve
        pval: P-value from normal approximation
        padj: FDR-adjusted p-value (Benjamini-Hochberg)
        pct_in: Percentage of cells expressing in group
        pct_out: Percentage of cells expressing out of group
        comparison: Comparison label (e.g., "group1_vs_group2")
    """

    feature: int
    group: str
    avg_expr: float
    log_fc: float
    ustat: float
    auc: float
    pval: float
    padj: float
    pct_in: float
    pct_out: float
    comparison: str

    def __repr__(self) -> str:
        """String representation of result."""
        return (
            f"WilcoxResult(feature={self.feature}, group='{self.group}', "
            f"log_fc={self.log_fc:.3f}, auc={self.auc:.3f}, "
            f"pval={self.pval:.3e}, padj={self.padj:.3e})"
        )
