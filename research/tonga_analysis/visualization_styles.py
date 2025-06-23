"""
Visualization styles module for consistent styling across all analyses.
This module provides color schemes and styling functions for visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Regional comparison color scheme
REGION_COLORS = {
    'tonga': '#E63946',  # Red
    'fiji': '#1D3557',   # Navy Blue
    'samoa': '#457B9D',  # Blue
    'tahiti': '#43AA8B'  # Teal
}

# Tonga islands color scheme
ISLAND_COLORS = {
    'tongatapu': '#FFB703',  # Yellow/Gold
    'vava\'u': '#219EBC',    # Blue
    'ha\'apai': '#8ECAE6',   # Light Blue
    '\'eua': '#90BE6D',      # Green
    'niuas': '#F8961E'       # Orange
}

# Ensure lowercase keys are also available
ISLAND_COLORS_LOWER = {k.lower(): v for k, v in ISLAND_COLORS.items()}

# Sentiment color scheme
SENTIMENT_COLORS = {
    'positive': '#57A773',  # Green
    'neutral': '#4A6FA5',   # Blue
    'negative': '#C33C54'   # Red
}

# Standard sentiment scale for heatmaps
# Values less than 0.25 are reddish (negative)
# Values between 0.25 and 0.35 are yellowish (neutral)
# Values greater than 0.35 are greenish (positive)
SENTIMENT_MIN = 0.0
SENTIMENT_MID_LOW = 0.25
SENTIMENT_MID_HIGH = 0.35
SENTIMENT_MAX = 0.5  # Most sentiment scores won't exceed 0.5, but we set the max to ensure good color contrast

# Additional categorical color palettes
CATEGORICAL_PALETTE = sns.color_palette('Set2', 8)


def set_visualization_style():
    """Set the default visualization style for all plots."""
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 22


def get_regional_palette(df, column='country'):
    """
    Get color palette for regional comparisons.
    
    Args:
        df: DataFrame containing country data
        column: Name of the column containing country names
        
    Returns:
        List of colors corresponding to countries in the DataFrame
    """
    # Convert country names to lowercase for consistent mapping
    countries = [c.lower() for c in df[column].unique()]
    return [REGION_COLORS.get(c, '#AAAAAA') for c in countries]


def get_island_palette(df, column='island'):
    """
    Get color palette for Tonga island comparisons.
    
    Args:
        df: DataFrame containing island data
        column: Name of the column containing island names
        
    Returns:
        List of colors corresponding to islands in the DataFrame
    """
    # Convert island names to lowercase for consistent mapping
    islands = [i.lower() for i in df[column].unique()]
    return [ISLAND_COLORS_LOWER.get(i, '#AAAAAA') for i in islands]


def get_sentiment_palette(include_neutral=True):
    """
    Get color palette for sentiment analysis.
    
    Args:
        include_neutral: Whether to include neutral category
        
    Returns:
        List of colors for sentiment categories
    """
    if include_neutral:
        return [SENTIMENT_COLORS['positive'], SENTIMENT_COLORS['neutral'], SENTIMENT_COLORS['negative']]
    else:
        return [SENTIMENT_COLORS['positive'], SENTIMENT_COLORS['negative']]
        
def get_standard_sentiment_cmap():
    """
    Get a standardized colormap for sentiment heatmaps.
    
    This returns a colormap that consistently maps:
    - Values less than 0.25 to reddish colors (lower sentiment)
    - Values between 0.25 and 0.35 to yellowish colors (medium sentiment)
    - Values above 0.35 to greenish colors (higher sentiment)
    
    Returns:
        Matplotlib colormap suitable for sentiment heatmaps with fixed value ranges
    """
    # Use RdYlGn colormap (Red-Yellow-Green)
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    # Create a custom colormap with fixed value ranges
    cmap = cm.get_cmap('RdYlGn')
    
    # Create a normalization function that maps our sentiment ranges to colormap values (0-1)
    norm = mcolors.Normalize(vmin=SENTIMENT_MIN, vmax=SENTIMENT_MAX)
    
    return cmap, norm


def apply_regional_style(ax, title=None, x_label=None, y_label=None):
    """
    Apply consistent styling to a plot with regional data.
    
    Args:
        ax: Matplotlib axes object
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
    """
    if title:
        ax.set_title(title, fontweight='bold', pad=15, fontsize=22)
    if x_label:
        ax.set_xlabel(x_label, fontsize=20)
    if y_label:
        ax.set_ylabel(y_label, fontsize=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Capitalize legend labels for countries
    if ax.get_legend():
        for t in ax.get_legend().texts:
            t.set_text(t.get_text().capitalize())
        ax.get_legend().set_title('Country', prop={'size': 16})
        plt.setp(ax.get_legend().get_texts(), fontsize=14)


def apply_island_style(ax, title=None, x_label=None, y_label=None):
    """
    Apply consistent styling to a plot with island data.
    
    Args:
        ax: Matplotlib axes object
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
    """
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    # Format island names in legend if present
    if ax.get_legend():
        for t in ax.get_legend().texts:
            text = t.get_text()
            if text.lower() in ['tongatapu', 'vavau', "vava'u", 'haapai', "ha'apai", 'eua', "'eua", 'niuas']:
                if text.lower() in ['tongatapu']:
                    t.set_text('Tongatapu')
                elif text.lower() in ['vavau', "vava'u"]:
                    t.set_text("Vava'u")
                elif text.lower() in ['haapai', "ha'apai"]:
                    t.set_text("Ha'apai")
                elif text.lower() in ['eua', "'eua"]:
                    t.set_text("'Eua")
                elif text.lower() in ['niuas']:
                    t.set_text('Niuas')