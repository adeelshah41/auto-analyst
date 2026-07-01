import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(4)

def find_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list:
    """Find pairs of columns with strong correlations (|r| > threshold).
    Returns list of dicts with col1, col2, correlation, strength."""
    if corr_matrix.empty:
        return []
    strong = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                strength = 'very strong' if abs(r) >= 0.9 else 'strong'
                direction = 'positive' if r > 0 else 'negative'
                strong.append({
                    'col1': cols[i],
                    'col2': cols[j],
                    'correlation': round(r, 4),
                    'strength': strength,
                    'direction': direction
                })
    return sorted(strong, key=lambda x: abs(x['correlation']), reverse=True)

def detect_outliers(df: pd.DataFrame) -> dict:
    """Detect outliers using IQR method for each numeric column.
    Returns dict with column name -> outlier info."""
    numeric_df = df.select_dtypes(include='number')
    outliers = {}
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        if len(data) == 0:
            continue
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (data < lower) | (data > upper)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            outliers[col] = {
                'count': outlier_count,
                'percentage': round(outlier_count / len(data) * 100, 2),
                'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4),
                'min_outlier': round(float(data[outlier_mask].min()), 4),
                'max_outlier': round(float(data[outlier_mask].max()), 4)
            }
    return outliers

def detect_trends(df: pd.DataFrame, time_col: str, value_col: str) -> dict:
    """Detect linear trend in a time series using linear regression.
    Returns trend direction, slope, R-squared, and p-value."""
    if time_col not in df.columns or value_col not in df.columns:
        return {}
    clean = df[[time_col, value_col]].dropna()
    if len(clean) < 3:
        return {}
    x = clean[time_col].values.astype(float)
    y = clean[value_col].values.astype(float)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    return {
        'direction': direction,
        'slope': round(slope, 6),
        'r_squared': round(r_value ** 2, 4),
        'p_value': round(p_value, 6),
        'significant': p_value < 0.05,
        'interpretation': f"{value_col} is {direction} over time" + 
                         (f" (statistically significant, p={p_value:.4f})" if p_value < 0.05 else f" (not statistically significant, p={p_value:.4f})")
    }

def perform_clustering(df: pd.DataFrame, n_clusters: int = 3, max_features: int = 10) -> dict:
    """Perform K-Means clustering on numeric features.
    Returns cluster labels, centers, and PCA-reduced coordinates for visualization."""
    numeric_df = df.select_dtypes(include='number').dropna()
    if numeric_df.shape[0] < n_clusters or numeric_df.shape[1] < 2:
        return {}
    
    # Limit features
    feature_cols = numeric_df.columns[:max_features].tolist()
    X = numeric_df[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Cluster summaries
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    
    return {
        'labels': labels,
        'pca_coords': X_pca,
        'feature_cols': feature_cols,
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'explained_variance': round(sum(pca.explained_variance_ratio_) * 100, 2),
        'inertia': round(kmeans.inertia_, 4)
    }

def compute_anova(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """Perform one-way ANOVA test between groups."""
    if group_col not in df.columns or value_col not in df.columns:
        return {}
    groups = [group[value_col].dropna().values for name, group in df.groupby(group_col) if len(group[value_col].dropna()) > 1]
    if len(groups) < 2:
        return {}
    f_stat, p_value = scipy_stats.f_oneway(*groups)
    return {
        'f_statistic': round(float(f_stat), 4),
        'p_value': round(float(p_value), 6),
        'significant': p_value < 0.05,
        'interpretation': f"Significant difference in {value_col} across {group_col} groups" if p_value < 0.05 else f"No significant difference in {value_col} across {group_col} groups"
    }

def get_all_patterns(df: pd.DataFrame, dataset_name: str = '') -> dict:
    """Run all pattern detection analyses and return combined results."""
    results = {}
    
    # Correlations
    corr_matrix = compute_correlations(df)
    results['correlation_matrix'] = corr_matrix
    results['strong_correlations'] = find_strong_correlations(corr_matrix)
    
    # Outliers
    results['outliers'] = detect_outliers(df)
    
    # Clustering
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 2:
        n_clusters = min(4, max(2, len(df) // 50))
        n_clusters = max(2, min(n_clusters, 5))
        results['clustering'] = perform_clustering(df, n_clusters=n_clusters)
    else:
        results['clustering'] = {}
    
    # Trends (for datasets with time columns)
    time_cols = [c for c in ['year', 'Year'] if c in df.columns]
    if time_cols:
        time_col = time_cols[0]
        numeric_cols_list = df.select_dtypes(include='number').columns.tolist()
        numeric_cols_list = [c for c in numeric_cols_list if c.lower() != 'year']
        trends = {}
        for col in numeric_cols_list:
            # Aggregate by time column first
            agg_df = df.groupby(time_col)[col].mean().reset_index()
            trend = detect_trends(agg_df, time_col, col)
            if trend:
                trends[col] = trend
        results['trends'] = trends
    else:
        results['trends'] = {}
    
    return results
