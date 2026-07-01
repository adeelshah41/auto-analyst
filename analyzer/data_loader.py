import pandas as pd
import plotly.express as px
import os

DATASET_INFO = {
    'Gapminder': {
        'icon': '🌍',
        'description': 'Country-level life expectancy, GDP per capita, and population data spanning 1952-2007.',
        'source': 'Plotly Express (built-in)',
        'kaggle_url': None
    },
    'Tips': {
        'icon': '🍽️',
        'description': 'Restaurant tipping behavior data including bill amount, tip, party size, and demographics.',
        'source': 'Plotly Express (built-in)',
        'kaggle_url': None
    },
    'World Happiness': {
        'icon': '😊',
        'description': 'World Happiness Report scores with contributing factors like GDP, social support, and freedom.',
        'source': 'Kaggle (World Happiness Report)',
        'kaggle_url': 'https://www.kaggle.com/datasets/unsdsn/world-happiness'
    }
}

def get_dataset_names() -> list:
    """Return list of available dataset names."""
    return list(DATASET_INFO.keys())

def get_dataset_info(name: str) -> dict:
    """Return metadata about a dataset."""
    return DATASET_INFO.get(name, {})

def load_gapminder() -> pd.DataFrame:
    """Load Gapminder dataset from plotly."""
    df = px.data.gapminder()
    return df

def load_tips() -> pd.DataFrame:
    """Load Tips dataset from plotly."""
    df = px.data.tips()
    # Add a tip_percentage column
    df['tip_pct'] = (df['tip'] / df['total_bill'] * 100).round(2)
    return df

def load_world_happiness() -> pd.DataFrame:
    """Load World Happiness Report from bundled CSV."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'world_happiness.csv')
    df = pd.read_csv(csv_path)
    return df

def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset by name. Returns a pandas DataFrame."""
    loaders = {
        'Gapminder': load_gapminder,
        'Tips': load_tips,
        'World Happiness': load_world_happiness
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loaders[name]()

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Return list of numeric column names."""
    return df.select_dtypes(include='number').columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    """Return list of categorical/object column names."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()
