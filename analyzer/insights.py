import os
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()


def generate_groq_insights(df: pd.DataFrame, patterns: dict, stats_summary: dict, dataset_name: str) -> str:
    """Generate insights using GROQ API. Falls back to rule-based if GROQ fails."""
    try:
        from groq import Groq
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        client = Groq(api_key=api_key)
        
        # Build context from analysis results
        context = _build_analysis_context(df, patterns, stats_summary, dataset_name)
        
        prompt = f"""You are an expert data analyst. Analyze the following statistical findings from the '{dataset_name}' dataset and provide a comprehensive, insightful executive summary.

Dataset Overview:
{context}

Provide your analysis in the following format:

## 🎯 Executive Summary
A brief 2-3 sentence overview of the most important findings.

## 📊 Key Findings
List 4-6 specific, data-driven insights. Each insight should:
- Reference specific numbers/statistics
- Explain WHY the finding matters
- Use bullet points with clear headers

## 🔍 Patterns & Relationships
Describe the most interesting correlations, trends, or clusters found.

## ⚠️ Anomalies & Outliers
Highlight any unusual data points or unexpected patterns.

## 💡 Recommendations
Provide 2-3 actionable recommendations based on the data.

Be specific, use actual numbers from the data, and make insights actionable."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert data analyst who provides clear, insightful, and actionable analysis. Use markdown formatting. Be specific and reference actual numbers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        
        insights_text = response.choices[0].message.content
        return insights_text, 'groq'
    
    except Exception as e:
        # Fall back to rule-based insights
        print(f"GROQ API failed: {e}. Falling back to rule-based insights.")
        return generate_rule_based_insights(df, patterns, stats_summary, dataset_name), 'rule-based'


def _build_analysis_context(df: pd.DataFrame, patterns: dict, stats_summary: dict, dataset_name: str) -> str:
    """Build a text context from analysis results for the LLM prompt."""
    lines = []
    
    # Basic info
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append("")
    
    # Descriptive stats (condensed)
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        lines.append("Descriptive Statistics:")
        for col in numeric_df.columns:
            lines.append(f"  {col}: mean={numeric_df[col].mean():.4f}, std={numeric_df[col].std():.4f}, min={numeric_df[col].min():.4f}, max={numeric_df[col].max():.4f}")
        lines.append("")
    
    # Strong correlations
    strong_corr = patterns.get('strong_correlations', [])
    if strong_corr:
        lines.append("Strong Correlations Found:")
        for c in strong_corr[:5]:
            lines.append(f"  {c['col1']} ↔ {c['col2']}: r={c['correlation']} ({c['strength']} {c['direction']})")
        lines.append("")
    
    # Outliers
    outliers = patterns.get('outliers', {})
    if outliers:
        lines.append("Outliers Detected:")
        for col, info in list(outliers.items())[:5]:
            lines.append(f"  {col}: {info['count']} outliers ({info['percentage']}% of data)")
        lines.append("")
    
    # Trends
    trends = patterns.get('trends', {})
    if trends:
        lines.append("Time Trends:")
        for col, trend in trends.items():
            lines.append(f"  {col}: {trend['direction']} (R²={trend['r_squared']}, p={trend['p_value']})")
        lines.append("")
    
    # Clustering
    clustering = patterns.get('clustering', {})
    if clustering:
        lines.append(f"Clustering: {clustering.get('n_clusters', 0)} clusters found")
        lines.append(f"  Explained variance (PCA 2D): {clustering.get('explained_variance', 0)}%")
        sizes = clustering.get('cluster_sizes', {})
        for k, v in sizes.items():
            lines.append(f"  Cluster {k}: {v} items")
    
    return '\n'.join(lines)


def generate_rule_based_insights(df: pd.DataFrame, patterns: dict, stats_summary: dict, dataset_name: str) -> str:
    """Generate insights using rule-based templates. Used as fallback when GROQ is unavailable."""
    insights = []
    
    # --- Executive Summary ---
    insights.append("## 🎯 Executive Summary")
    insights.append(f"Analysis of the **{dataset_name}** dataset ({df.shape[0]:,} rows × {df.shape[1]} columns) ")
    
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    insights.append(f"reveals patterns across {len(numeric_cols)} numeric and {len(cat_cols)} categorical variables.\n")
    
    # --- Key Findings ---
    insights.append("## 📊 Key Findings")
    
    # Top stats insights
    for col in numeric_cols[:4]:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        mean_val = data.mean()
        std_val = data.std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0
        
        if cv > 100:
            badge = "🔴"
            variability = "extremely high variability"
        elif cv > 50:
            badge = "🟡"
            variability = "high variability"
        else:
            badge = "🟢"
            variability = "moderate variability"
        
        insights.append(f"- {badge} **{col}**: Mean = {mean_val:,.4f}, Std = {std_val:,.4f} "
                       f"(CV = {cv:.1f}%, {variability})")
    insights.append("")
    
    # --- Correlations ---
    strong_corr = patterns.get('strong_correlations', [])
    if strong_corr:
        insights.append("## 🔗 Patterns & Relationships")
        for c in strong_corr[:5]:
            emoji = "📈" if c['direction'] == 'positive' else "📉"
            insights.append(f"- {emoji} **{c['col1']}** and **{c['col2']}** have a "
                          f"{c['strength']} {c['direction']} correlation (r = {c['correlation']}).")
            if c['direction'] == 'positive':
                insights.append(f"  *As {c['col1']} increases, {c['col2']} tends to increase as well.*")
            else:
                insights.append(f"  *As {c['col1']} increases, {c['col2']} tends to decrease.*")
        insights.append("")
    
    # --- Outliers ---
    outliers = patterns.get('outliers', {})
    if outliers:
        insights.append("## ⚠️ Anomalies & Outliers")
        for col, info in list(outliers.items())[:5]:
            severity = "🔴" if info['percentage'] > 10 else "🟡" if info['percentage'] > 5 else "🟢"
            insights.append(f"- {severity} **{col}**: {info['count']} outliers detected "
                          f"({info['percentage']}% of data). "
                          f"Range: [{info['lower_bound']:,.4f}, {info['upper_bound']:,.4f}]. "
                          f"Extreme values: {info['min_outlier']:,.4f} to {info['max_outlier']:,.4f}.")
        insights.append("")
    
    # --- Trends ---
    trends = patterns.get('trends', {})
    if trends:
        insights.append("## 📈 Time Trends")
        for col, trend in trends.items():
            emoji = "⬆️" if trend['direction'] == 'increasing' else "⬇️" if trend['direction'] == 'decreasing' else "➡️"
            sig = "✅ statistically significant" if trend['significant'] else "❌ not statistically significant"
            insights.append(f"- {emoji} **{col}** is **{trend['direction']}** over time "
                          f"(R² = {trend['r_squared']}, {sig}, p = {trend['p_value']})")
        insights.append("")
    
    # --- Clustering ---
    clustering = patterns.get('clustering', {})
    if clustering:
        insights.append("## 🔬 Cluster Analysis")
        n = clustering.get('n_clusters', 0)
        ev = clustering.get('explained_variance', 0)
        insights.append(f"K-Means clustering identified **{n} natural groupings** in the data.")
        insights.append(f"The first two principal components explain **{ev}%** of the variance.")
        sizes = clustering.get('cluster_sizes', {})
        for k, v in sizes.items():
            insights.append(f"- Cluster {k}: {v} records")
        insights.append("")
    
    # --- Data Quality ---
    if stats_summary:
        insights.append("## 🏥 Data Quality")
        insights.append(f"- Total records: **{stats_summary.get('total_rows', 0):,}**")
        insights.append(f"- Missing values: **{stats_summary.get('missing_pct', 0)}%**")
        insights.append(f"- Duplicate rows: **{stats_summary.get('duplicate_rows', 0)}**")
        insights.append(f"- Memory usage: **{stats_summary.get('memory_mb', 0)} MB**")
    
    return '\n'.join(insights)
