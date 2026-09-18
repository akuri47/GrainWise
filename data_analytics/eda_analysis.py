"""
Exploratory Data Analysis (EDA) script for Rice Grain Quality Analyzer.

This script loads or generates grain morphological feature data and produces 
various plots and summary statistics to understand the data distribution.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'grain_features.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

FEATURES = [
    'length_mm', 'width_mm', 'aspect_ratio', 'area_mm2',
    'circularity', 'solidity', 'eccentricity',
    'rg_ratio', 'rb_ratio', 'gb_ratio', 'norm_s'
]

VARIETIES = [
    'swarna', '1010', '1001', 'ganga_kaveri', 'mansuri', 
    'golden_mansuri', 'nati_mansuri', 'sonam', 'ir64'
]

DEFECTS = ['whole', 'broken', 'chalky', 'damaged', 'discolored', 'foreign']

def generate_synthetic_data(num_samples=2000):
    """Generates synthetic rice grain data for testing."""
    print("Generating synthetic data...")
    np.random.seed(42)
    
    data = []
    for _ in range(num_samples):
        variety = np.random.choice(VARIETIES)
        defect = np.random.choice(DEFECTS, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        
        # Base dimensions (variety dependent)
        if variety == 'sonam':
            l_mean, w_mean = 6.5, 1.8
        elif variety == 'swarna':
            l_mean, w_mean = 5.0, 2.2
        elif variety == 'ir64':
            l_mean, w_mean = 6.0, 2.0
        else:
            l_mean, w_mean = np.random.uniform(5.2, 6.2), np.random.uniform(1.9, 2.1)
            
        length = np.random.normal(l_mean, 0.2)
        width = np.random.normal(w_mean, 0.1)
        
        # Ensure physical constraints
        length = max(length, width + 0.5)
        aspect_ratio = length / width
        area = length * width * 0.8
        
        # Other morphological features
        circularity = np.random.normal(0.6, 0.05)
        solidity = np.random.normal(0.95, 0.02)
        eccentricity = np.random.normal(0.85, 0.03)
        
        # Color features
        rg_ratio = np.random.normal(1.05, 0.05)
        rb_ratio = np.random.normal(1.2, 0.1)
        gb_ratio = np.random.normal(1.15, 0.08)
        norm_s = np.random.normal(0.1, 0.02)
        
        data.append({
            'variety': variety,
            'defect_type': defect,
            'length_mm': length,
            'width_mm': width,
            'aspect_ratio': aspect_ratio,
            'area_mm2': area,
            'circularity': circularity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'rg_ratio': rg_ratio,
            'rb_ratio': rb_ratio,
            'gb_ratio': gb_ratio,
            'norm_s': norm_s
        })
        
    return pd.DataFrame(data)

def load_data():
    """Loads data from CSV or generates synthetic data if not found."""
    if os.path.exists(DATA_FILE):
        print(f"Loading data from {DATA_FILE}")
        return pd.read_csv(DATA_FILE)
    else:
        print(f"Data file not found at {DATA_FILE}")
        return generate_synthetic_data()

def setup_plotting():
    """Configures matplotlib and seaborn aesthetics."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 150

def generate_plots(df):
    """Generates all required EDA plots and saves them to disk."""
    print("Generating plots...")
    
    # 1. Correlation Heatmap
    print(" - Generating correlation heatmap...")
    plt.figure()
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()
    
    # Boxplots by Variety
    boxplots = {
        'length_mm': 'boxplot_length_by_variety.png',
        'width_mm': 'boxplot_width_by_variety.png',
        'area_mm2': 'boxplot_area_by_variety.png',
        'aspect_ratio': 'boxplot_aspect_ratio_by_variety.png'
    }
    
    for feature, filename in boxplots.items():
        print(f" - Generating boxplot for {feature}...")
        plt.figure()
        sns.boxplot(data=df, x='variety', y=feature)
        plt.title(f"{feature} by Variety")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.close()
        
    # Grain count by variety
    print(" - Generating grain count distribution...")
    plt.figure()
    sns.countplot(data=df, x='variety', order=df['variety'].value_counts().index)
    plt.title("Grain Count by Variety")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'grain_count_distribution.png'))
    plt.close()
    
    # Defect distribution pie chart
    print(" - Generating defect distribution pie chart...")
    plt.figure()
    defect_counts = df['defect_type'].value_counts()
    plt.pie(defect_counts, labels=defect_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Defect Category Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'defect_distribution_pie.png'))
    plt.close()
    
    # Length vs Width scatter
    print(" - Generating length vs width scatter plot...")
    plt.figure()
    sns.scatterplot(data=df, x='length_mm', y='width_mm', hue='variety', alpha=0.7)
    plt.title("Length vs Width colored by Variety")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'length_vs_width_scatter.png'))
    plt.close()

def save_summary_statistics(df):
    """Calculates and saves summary statistics grouped by variety."""
    print("Saving summary statistics...")
    summary = df.groupby('variety')[FEATURES].agg(['mean', 'std', 'min', 'max', 'median'])
    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    output_file = os.path.join(OUTPUTS_DIR, 'summary_statistics.csv')
    summary.to_csv(output_file)
    print(f"Summary statistics saved to {output_file}")

def main():
    print("Starting EDA Analysis...")
    
    # Ensure directories exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    setup_plotting()
    
    df = load_data()
    
    generate_plots(df)
    save_summary_statistics(df)
    
    print("EDA Analysis completed successfully.")

if __name__ == '__main__':
    main()
