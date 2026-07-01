"""
The Auto-Analyst — AI-Powered Data Insights Engine
Main Streamlit application entry point.
Automatically interprets data, surfaces patterns, and writes AI-powered insights.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration — MUST be first Streamlit command
st.set_page_config(
    page_title="The Auto-Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import project modules
from analyzer.data_loader import load_dataset, get_dataset_names, get_dataset_info, get_numeric_columns, get_categorical_columns
from analyzer.statistics import compute_descriptive_stats, compute_data_quality, compute_group_stats, compute_categorical_summary
from analyzer.patterns import get_all_patterns, compute_correlations, find_strong_correlations
from analyzer.insights import generate_groq_insights, generate_rule_based_insights
from visualizer.charts import (
    correlation_heatmap, distribution_histogram, box_plot,
    scatter_plot, line_chart, cluster_scatter, bar_chart
)


# ─────────────────────────────────────────────────────────────────
# Custom CSS for premium dark theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #121620 50%, #0E1117 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141821 0%, #1A1D29 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6C63FF, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8B8FA3;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1A1D29 0%, #1E2235 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        border-color: rgba(108, 99, 255, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.15);
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #6C63FF;
        line-height: 1.2;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8B8FA3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.4rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(26, 29, 41, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        color: #8B8FA3;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF20, #4ECDC420) !important;
        color: #FFFFFF !important;
    }

    /* Insight cards */
    .insight-card {
        background: linear-gradient(145deg, #1A1D29, #1E2235);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* Badge styles */
    .badge-groq {
        background: linear-gradient(135deg, #6C63FF, #4ECDC4);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-fallback {
        background: linear-gradient(135deg, #FFA07A, #FF6B6B);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Dataset info card */
    .dataset-info {
        background: linear-gradient(145deg, #1A1D29, #1E2235);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 14px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .dataset-info h4 {
        color: #4ECDC4;
        margin: 0 0 0.3rem 0;
    }
    .dataset-info p {
        color: #8B8FA3;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Streamlit elements overrides */
    .stSelectbox label, .stMultiselect label, .stSlider label {
        color: #C0C0D0 !important;
        font-weight: 500 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #5A52E0);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7B73FF, #6C63FF);
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
        transform: translateY(-1px);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(26, 29, 41, 0.5);
        border-radius: 10px;
        color: #E0E0E0 !important;
    }

    /* Spinner override */
    .stSpinner > div > div {
        border-top-color: #6C63FF !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 The Auto-Analyst</h1>
    <p>AI-powered data interpretation • Pattern detection • Automated insights</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Sidebar — Dataset Selection & Filters
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Dataset Selection")

    dataset_names = get_dataset_names()
    selected_dataset = st.selectbox(
        "Choose a dataset",
        dataset_names,
        index=0,
        help="Select one of the pre-loaded datasets to analyze"
    )

    # Dataset info card
    info = get_dataset_info(selected_dataset)
    st.markdown(f"""
    <div class="dataset-info">
        <h4>{info['icon']} {selected_dataset}</h4>
        <p>{info['description']}</p>
        <p style="margin-top: 6px; font-size: 0.78rem; color: #6C63FF;">Source: {info['source']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Load the dataset
    @st.cache_data(show_spinner=False)
    def cached_load(name):
        return load_dataset(name)

    raw_df = cached_load(selected_dataset)
    df = raw_df.copy()

    # Dataset-specific filters
    st.markdown("### 🔧 Filters")

    if selected_dataset == 'Gapminder':
        continents = sorted(df['continent'].unique())
        selected_continents = st.multiselect(
            "Continents", continents, default=continents,
            help="Filter by continent"
        )
        if selected_continents:
            df = df[df['continent'].isin(selected_continents)]

        year_range = st.slider(
            "Year Range",
            int(df['year'].min()), int(df['year'].max()),
            (int(df['year'].min()), int(df['year'].max())),
            step=5
        )
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

        countries = sorted(df['country'].unique())
        selected_countries = st.multiselect(
            "Countries (optional)", countries, default=[],
            help="Leave empty to include all countries"
        )
        if selected_countries:
            df = df[df['country'].isin(selected_countries)]

    elif selected_dataset == 'Tips':
        days = sorted(df['day'].unique(), key=lambda x: ['Thur', 'Fri', 'Sat', 'Sun'].index(x) if x in ['Thur', 'Fri', 'Sat', 'Sun'] else 0)
        selected_days = st.multiselect("Day", days, default=days)
        if selected_days:
            df = df[df['day'].isin(selected_days)]

        times = df['time'].unique().tolist()
        selected_times = st.multiselect("Meal Time", times, default=times)
        if selected_times:
            df = df[df['time'].isin(selected_times)]

        smoker_filter = st.selectbox("Smoker", ["All", "Yes", "No"])
        if smoker_filter != "All":
            df = df[df['smoker'] == smoker_filter]

    elif selected_dataset == 'World Happiness':
        if 'Region' in df.columns:
            regions = sorted(df['Region'].unique())
            selected_regions = st.multiselect("Region", regions, default=regions)
            if selected_regions:
                df = df[df['Region'].isin(selected_regions)]

        if 'Year' in df.columns:
            years = sorted(df['Year'].unique())
            selected_years = st.multiselect("Year", years, default=[max(years)])
            if selected_years:
                df = df[df['Year'].isin(selected_years)]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Filtered data summary
    st.markdown(f"**📋 Filtered Data:** {len(df):,} rows × {len(df.columns)} columns")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Generate insights button
    st.markdown("### 💡 AI Insights")
    generate_insights = st.button("🚀 Generate AI Insights", use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# Main Content — Tabs
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Analysis",
    "🔗 Correlations",
    "💡 AI Insights"
])


# ─────────── TAB 1: OVERVIEW ─────────────────────────────────────
with tab1:
    # Data Quality Metrics
    quality = compute_data_quality(df)

    cols = st.columns(4)
    metrics = [
        ("Rows", f"{quality['total_rows']:,}"),
        ("Columns", f"{quality['total_columns']}"),
        ("Missing", f"{quality['missing_pct']}%"),
        ("Duplicates", f"{quality['duplicate_rows']:,}")
    ]
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Descriptive Statistics
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📐 Descriptive Statistics")
        desc_stats = compute_descriptive_stats(df)
        st.dataframe(desc_stats, use_container_width=True, height=350)

    with col_right:
        st.markdown("#### 🏷️ Categorical Summary")
        cat_summary = compute_categorical_summary(df)
        if cat_summary:
            for col_name, info in cat_summary.items():
                with st.expander(f"**{col_name}** — {info['unique_count']} unique values"):
                    st.write(f"**Most common:** {info['top_value']} ({info['top_count']} occurrences)")
                    dist_df = pd.DataFrame(
                        list(info['distribution'].items()),
                        columns=['Value', 'Count']
                    )
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns in the current view.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Data Preview
    st.markdown("#### 🔍 Data Preview")
    preview_tab1, preview_tab2 = st.tabs(["First Rows", "Last Rows"])
    with preview_tab1:
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with preview_tab2:
        st.dataframe(df.tail(15), use_container_width=True, hide_index=True)


# ─────────── TAB 2: ANALYSIS ─────────────────────────────────────
with tab2:
    numeric_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    if not numeric_cols:
        st.warning("No numeric columns available for analysis.")
    else:
        # Distribution Analysis
        st.markdown("#### 📊 Distribution Analysis")

        dist_col1, dist_col2 = st.columns([1, 1])
        with dist_col1:
            selected_num_col = st.selectbox("Select numeric column", numeric_cols, key="dist_col")
        with dist_col2:
            color_by = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="dist_color")

        color_col = color_by if color_by != "None" else None
        fig_hist = distribution_histogram(df, selected_num_col, color_col=color_col)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Box Plot Comparison
        st.markdown("#### 📦 Box Plot Comparison")
        box_col1, box_col2 = st.columns([1, 1])
        with box_col1:
            box_value = st.selectbox("Value column", numeric_cols, key="box_val")
        with box_col2:
            box_group = st.selectbox("Group by", ["None"] + cat_cols, key="box_group")

        group_col = box_group if box_group != "None" else None
        fig_box = box_plot(df, box_value, group_col=group_col)
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Time Series (if applicable)
        time_cols = [c for c in ['year', 'Year'] if c in df.columns]
        if time_cols:
            st.markdown("#### 📈 Trends Over Time")
            time_col = time_cols[0]

            trend_col1, trend_col2 = st.columns([1, 1])
            with trend_col1:
                trend_metric = st.selectbox(
                    "Metric",
                    [c for c in numeric_cols if c.lower() != 'year'],
                    key="trend_metric"
                )
            with trend_col2:
                if cat_cols:
                    trend_group = st.selectbox("Group by", ["Overall Mean"] + cat_cols, key="trend_group")
                else:
                    trend_group = "Overall Mean"

            if trend_group == "Overall Mean":
                trend_df = df.groupby(time_col)[trend_metric].mean().reset_index()
                fig_line = line_chart(trend_df, time_col, trend_metric)
            else:
                trend_df = df.groupby([time_col, trend_group])[trend_metric].mean().reset_index()
                fig_line = line_chart(trend_df, time_col, trend_metric, color_col=trend_group)

            st.plotly_chart(fig_line, use_container_width=True)

        # Scatter Plot Explorer
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🔍 Scatter Plot Explorer")

        scatter_c1, scatter_c2, scatter_c3 = st.columns([1, 1, 1])
        with scatter_c1:
            scatter_x = st.selectbox("X-axis", numeric_cols, index=0, key="sc_x")
        with scatter_c2:
            scatter_y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="sc_y")
        with scatter_c3:
            scatter_color = st.selectbox("Color by", ["None"] + cat_cols, key="sc_color")

        sc_color = scatter_color if scatter_color != "None" else None

        if scatter_x == scatter_y:
            st.info("⚠️ Please select different columns for the X and Y axes to generate a scatter plot.")
        else:
            # Determine hover column (use first string column with unique-ish values like country/name)
            hover_col = None
            for c in cat_cols:
                if df[c].nunique() > 5:
                    hover_col = c
                    break

            fig_scatter = scatter_plot(df, scatter_x, scatter_y, color_col=sc_color, hover_col=hover_col)
            st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────── TAB 3: CORRELATIONS ─────────────────────────────────
with tab3:
    numeric_cols = get_numeric_columns(df)

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        # Correlation Matrix
        corr_matrix = compute_correlations(df)
        fig_corr = correlation_heatmap(corr_matrix)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Strong Correlations
        strong = find_strong_correlations(corr_matrix)
        if strong:
            st.markdown("#### 🔗 Strong Correlations Detected")
            for item in strong:
                direction_emoji = "📈" if item['direction'] == 'positive' else "📉"
                st.markdown(f"""
                <div class="insight-card">
                    <strong>{direction_emoji} {item['col1']}</strong> ↔ <strong>{item['col2']}</strong><br>
                    <span style="color: #6C63FF; font-size: 1.3rem; font-weight: 700;">r = {item['correlation']}</span>
                    <span style="color: #8B8FA3; margin-left: 10px;">({item['strength']} {item['direction']})</span>
                </div>
                """, unsafe_allow_html=True)

                # Auto scatter plot for top correlated pairs
                fig_pair = scatter_plot(df, item['col1'], item['col2'],
                                       color_col=get_categorical_columns(df)[0] if get_categorical_columns(df) else None)
                st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.info("No strong correlations (|r| > 0.7) found in the current data.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Cluster Analysis
        st.markdown("#### 🔬 Cluster Analysis")
        patterns = get_all_patterns(df, selected_dataset)
        clustering = patterns.get('clustering', {})

        if clustering:
            fig_cluster = cluster_scatter(
                clustering.get('pca_coords'),
                clustering.get('labels'),
                clustering.get('explained_variance', 0)
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            # Cluster sizes
            sizes = clustering.get('cluster_sizes', {})
            if sizes:
                size_cols = st.columns(len(sizes))
                for i, (k, v) in enumerate(sizes.items()):
                    with size_cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="value">{v}</div>
                            <div class="label">Cluster {k}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Not enough numeric features for cluster analysis.")


# ─────────── TAB 4: AI INSIGHTS ──────────────────────────────────
with tab4:
    st.markdown("#### 💡 AI-Powered Data Insights")
    st.markdown("""
    <p style="color: #8B8FA3; margin-bottom: 1rem;">
        Click <strong>"Generate AI Insights"</strong> in the sidebar to analyze your data using GROQ AI. 
        The system automatically falls back to rule-based analysis if the AI service is unavailable.
    </p>
    """, unsafe_allow_html=True)

    # Compute patterns for insights
    @st.cache_data(show_spinner=False)
    def cached_patterns(_df_hash, df, dataset_name):
        return get_all_patterns(df, dataset_name)

    @st.cache_data(show_spinner=False)
    def cached_quality(_df_hash, df):
        return compute_data_quality(df)

    df_hash = pd.util.hash_pandas_object(df).sum()

    if generate_insights or 'insights_result' in st.session_state:
        if generate_insights:
            with st.spinner("🧠 Analyzing data and generating insights..."):
                patterns_result = cached_patterns(df_hash, df, selected_dataset)
                quality_result = cached_quality(df_hash, df)

                insights_text, mode = generate_groq_insights(
                    df, patterns_result, quality_result, selected_dataset
                )

                st.session_state['insights_result'] = insights_text
                st.session_state['insights_mode'] = mode
                st.session_state['insights_dataset'] = selected_dataset

        # Display insights
        mode = st.session_state.get('insights_mode', 'rule-based')
        insights_text = st.session_state.get('insights_result', '')
        insights_dataset = st.session_state.get('insights_dataset', '')

        # Mode badge
        if mode == 'groq':
            st.markdown('<span class="badge-groq">🤖 GROQ AI — llama-3.3-70b-versatile</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-fallback">⚙️ Rule-Based Analysis (Fallback)</span>', unsafe_allow_html=True)

        if insights_dataset != selected_dataset:
            st.warning(f"⚠️ These insights were generated for the **{insights_dataset}** dataset. Click 'Generate AI Insights' to update for **{selected_dataset}**.")

        st.markdown(f"""<div class="insight-card">{''}</div>""", unsafe_allow_html=True)
        st.markdown(insights_text)

    else:
        # Show placeholder with instructions
        st.markdown("""
        <div class="insight-card" style="text-align: center; padding: 3rem;">
            <p style="font-size: 3rem; margin-bottom: 1rem;">🧠</p>
            <h3 style="color: #6C63FF; margin-bottom: 0.5rem;">Ready to Analyze</h3>
            <p style="color: #8B8FA3;">
                Select a dataset, apply filters, then click <strong>"🚀 Generate AI Insights"</strong> in the sidebar 
                to get AI-powered analysis of your data.
            </p>
            <p style="color: #8B8FA3; font-size: 0.85rem; margin-top: 1rem;">
                <strong>Primary:</strong> GROQ AI (llama-3.3-70b-versatile)<br>
                <strong>Fallback:</strong> Rule-based statistical analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Always show quick stats summary
    with st.expander("📊 Quick Statistical Summary", expanded=False):
        patterns_quick = get_all_patterns(df, selected_dataset)

        # Trends
        trends = patterns_quick.get('trends', {})
        if trends:
            st.markdown("**📈 Detected Trends:**")
            for col, trend in trends.items():
                emoji = "⬆️" if trend['direction'] == 'increasing' else "⬇️" if trend['direction'] == 'decreasing' else "➡️"
                sig = "✅" if trend['significant'] else "❌"
                st.markdown(f"- {emoji} **{col}**: {trend['direction']} (R² = {trend['r_squared']}, significant: {sig})")

        # Outliers
        outliers = patterns_quick.get('outliers', {})
        if outliers:
            st.markdown("**⚠️ Outliers Detected:**")
            for col, info in outliers.items():
                st.markdown(f"- **{col}**: {info['count']} outliers ({info['percentage']}%)")

        # Strong correlations count
        strong_c = patterns_quick.get('strong_correlations', [])
        if strong_c:
            st.markdown(f"**🔗 Strong Correlations:** {len(strong_c)} pairs found")


# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #555; padding: 1rem 0;">
    <p style="font-size: 0.8rem;">
        🧠 <strong>The Auto-Analyst</strong> — AI-Powered Data Insights Engine<br>
        Built with Streamlit • Plotly • GROQ AI • scikit-learn<br>
        <span style="color: #6C63FF;">Datasets: Gapminder • Tips • World Happiness Report</span>
    </p>
</div>
""", unsafe_allow_html=True)
