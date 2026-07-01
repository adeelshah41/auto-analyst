import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

# Custom color palette
COLORS = {
    'primary': '#6C63FF',
    'secondary': '#FF6B6B',
    'teal': '#4ECDC4',
    'blue': '#45B7D1',
    'salmon': '#FFA07A',
    'mint': '#98D8C8',
    'gold': '#FFD93D',
    'purple': '#A855F7',
    'pink': '#EC4899',
    'emerald': '#10B981'
}

COLOR_SEQUENCE = list(COLORS.values())

LAYOUT_DEFAULTS = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#E0E0E0'),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(
        bgcolor='#1A1D29',
        font_size=13,
        font_family='Inter, sans-serif',
        bordercolor='#6C63FF'
    )
)


def _apply_defaults(fig, title: str = '', height: int = 450):
    """Apply consistent styling to all figures."""
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=18, color='#FFFFFF'), x=0.02),
        height=height,
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap."""
    if corr_matrix.empty:
        return go.Figure()
    
    # Custom colorscale: blue (negative) -> transparent (zero) -> purple (positive)
    colorscale = [
        [0.0, 'rgb(255,107,107)'],
        [0.25, 'rgb(180,80,80)'],
        [0.5, 'rgb(26,29,41)'],
        [0.75, 'rgb(80,73,180)'],
        [1.0, 'rgb(108,99,255)']
    ]
    
    text = corr_matrix.round(3).astype(str).values
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=11, color='white'),
        colorscale=colorscale,
        zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text='Correlation', font=dict(color='#E0E0E0')),
            tickfont=dict(color='#E0E0E0'),
            bgcolor='rgba(0,0,0,0)'
        ),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.4f}<extra></extra>'
    ))
    
    return _apply_defaults(fig, '🔗 Correlation Matrix', height=max(350, len(corr_matrix) * 45))


def distribution_histogram(df: pd.DataFrame, col: str, nbins: int = 30, color_col: str = None) -> go.Figure:
    """Create a histogram with optional KDE overlay."""
    if color_col and color_col in df.columns:
        fig = px.histogram(
            df, x=col, color=color_col, nbins=nbins,
            color_discrete_sequence=COLOR_SEQUENCE,
            opacity=0.75, barmode='overlay',
            marginal='box'
        )
    else:
        fig = px.histogram(
            df, x=col, nbins=nbins,
            color_discrete_sequence=[COLORS['primary']],
            opacity=0.8,
            marginal='box'
        )
    
    return _apply_defaults(fig, f'📊 Distribution of {col}')


def box_plot(df: pd.DataFrame, value_col: str, group_col: str = None) -> go.Figure:
    """Create a box plot, optionally grouped by a categorical column."""
    if group_col and group_col in df.columns:
        fig = px.box(
            df, x=group_col, y=value_col,
            color=group_col,
            color_discrete_sequence=COLOR_SEQUENCE,
            points='outliers'
        )
    else:
        fig = px.box(
            df, y=value_col,
            color_discrete_sequence=[COLORS['primary']],
            points='outliers'
        )
    
    return _apply_defaults(fig, f'📦 Box Plot: {value_col}' + (f' by {group_col}' if group_col else ''))


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, size_col: str = None, hover_col: str = None, trendline: bool = True) -> go.Figure:
    """Create an interactive scatter plot with optional trendline."""
    kwargs = dict(
        data_frame=df, x=x_col, y=y_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.7
    )
    if color_col and color_col in df.columns:
        kwargs['color'] = color_col
    if size_col and size_col in df.columns:
        kwargs['size'] = size_col
        kwargs['size_max'] = 30
    if hover_col and hover_col in df.columns:
        kwargs['hover_name'] = hover_col
    if trendline:
        try:
            import statsmodels  # noqa: F401
            kwargs['trendline'] = 'ols'
            kwargs['trendline_color_override'] = COLORS['secondary']
        except ImportError:
            pass  # Skip trendline if statsmodels not available
    
    fig = px.scatter(**kwargs)
    return _apply_defaults(fig, f'🔍 {x_col} vs {y_col}')


def line_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
    """Create a time-series line chart."""
    if color_col and color_col in df.columns:
        fig = px.line(
            df, x=x_col, y=y_col, color=color_col,
            color_discrete_sequence=COLOR_SEQUENCE,
            markers=True
        )
    else:
        fig = px.line(
            df, x=x_col, y=y_col,
            color_discrete_sequence=[COLORS['primary']],
            markers=True
        )
    
    fig.update_traces(line=dict(width=2))
    return _apply_defaults(fig, f'📈 {y_col} Over {x_col}')


def cluster_scatter(pca_coords, labels, explained_variance: float = 0) -> go.Figure:
    """Create a PCA scatter plot colored by cluster labels."""
    if pca_coords is None or labels is None:
        return go.Figure()
    
    cluster_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Cluster': [f'Cluster {l}' for l in labels]
    })
    
    fig = px.scatter(
        cluster_df, x='PC1', y='PC2', color='Cluster',
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.75
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='rgba(255,255,255,0.3)')))
    
    title = f'🔬 Cluster Analysis (PCA — {explained_variance}% variance explained)'
    return _apply_defaults(fig, title)


def bar_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, horizontal: bool = False) -> go.Figure:
    """Create a bar chart."""
    kwargs = dict(
        data_frame=df,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.85
    )
    if horizontal:
        kwargs.update(x=y_col, y=x_col)
    else:
        kwargs.update(x=x_col, y=y_col)
    
    if color_col and color_col in df.columns:
        kwargs['color'] = color_col
    
    fig = px.bar(**kwargs)
    return _apply_defaults(fig, f'📊 {y_col} by {x_col}')


def pie_chart(df: pd.DataFrame, names_col: str, values_col: str) -> go.Figure:
    """Create a donut/pie chart."""
    fig = px.pie(
        df, names=names_col, values=values_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        hole=0.4
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(size=12, color='white')
    )
    return _apply_defaults(fig, f'🍩 {values_col} Distribution by {names_col}')


def multi_line_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str = '') -> go.Figure:
    """Create a multi-line chart for comparing multiple metrics."""
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col],
            name=col, mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color)
        ))
    
    return _apply_defaults(fig, title or f'📈 Trends Over {x_col}')
