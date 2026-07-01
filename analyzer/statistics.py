import pandas as pd
import numpy as np
from scipy import stats

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for all numeric columns.
    Returns a DataFrame with mean, median, std, min, max, skewness, kurtosis."""
    numeric_df = df.select_dtypes(include='number')
    stats_dict = {
        'Mean': numeric_df.mean(),
        'Median': numeric_df.median(),
        'Std Dev': numeric_df.std(),
        'Min': numeric_df.min(),
        'Max': numeric_df.max(),
        'Skewness': numeric_df.skew(),
        'Kurtosis': numeric_df.kurtosis(),
        'Missing %': (numeric_df.isnull().sum() / len(numeric_df) * 100).round(2)
    }
    return pd.DataFrame(stats_dict).round(4)

def compute_data_quality(df: pd.DataFrame) -> dict:
    """Assess overall data quality."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    return {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'numeric_columns': len(df.select_dtypes(include='number').columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'total_missing': int(missing_cells),
        'missing_pct': round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0,
        'duplicate_rows': int(df.duplicated().sum()),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3)
    }

def compute_group_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Compute statistics grouped by a categorical column."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(group_col)[value_col].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    return grouped.round(4).sort_values('mean', ascending=False)

def compute_top_bottom(df: pd.DataFrame, col: str, n: int = 5) -> tuple:
    """Return top N and bottom N rows by a given column."""
    if col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    sorted_df = df.sort_values(col, ascending=False)
    return sorted_df.head(n), sorted_df.tail(n)

def compute_categorical_summary(df: pd.DataFrame) -> dict:
    """Compute value counts for each categorical column."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    summary = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        summary[col] = {
            'unique_count': int(vc.shape[0]),
            'top_value': str(vc.index[0]) if len(vc) > 0 else None,
            'top_count': int(vc.iloc[0]) if len(vc) > 0 else 0,
            'distribution': vc.head(10).to_dict()
        }
    return summary
