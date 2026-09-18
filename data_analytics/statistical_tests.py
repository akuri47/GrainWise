"""
Statistical testing script for the Rice Grain Quality Analyzer project.
Runs hypothesis tests on grain morphological data across 9 rice varieties.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Any

VARIETIES = ['swarna', '1010', '1001', 'ganga_kaveri', 'mansuri', 'golden_mansuri', 'nati_mansuri', 'sonam', 'ir64']
DEFECT_CLASSES = ['whole', 'broken', 'chalky', 'damaged', 'discolored', 'foreign']
FEATURES = [
    'length_mm', 'width_mm', 'aspect_ratio', 'area_mm2',
    'circularity', 'solidity', 'eccentricity',
    'rg_ratio', 'rb_ratio', 'gb_ratio', 'norm_s'
]
KEY_FEATURES = ['length_mm', 'width_mm', 'area_mm2', 'aspect_ratio']

def generate_synthetic_data(num_samples: int = 1000) -> pd.DataFrame:
    """Generate realistic synthetic data for rice grains."""
    np.random.seed(42)
    data = []
    
    # Base characteristics for varieties to create differences
    variety_params = {
        'swarna': {'l': 5.5, 'w': 2.2, 'a': 12.0},
        '1010': {'l': 6.0, 'w': 2.1, 'a': 12.5},
        '1001': {'l': 6.2, 'w': 2.3, 'a': 14.0},
        'ganga_kaveri': {'l': 6.5, 'w': 2.0, 'a': 13.0},
        'mansuri': {'l': 5.2, 'w': 2.4, 'a': 12.0},
        'golden_mansuri': {'l': 5.4, 'w': 2.3, 'a': 12.2},
        'nati_mansuri': {'l': 5.1, 'w': 2.5, 'a': 12.5},
        'sonam': {'l': 5.8, 'w': 2.1, 'a': 12.0},
        'ir64': {'l': 6.8, 'w': 2.2, 'a': 14.5},
    }
    
    for _ in range(num_samples):
        variety = np.random.choice(VARIETIES)
        defect = np.random.choice(DEFECT_CLASSES, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        
        vp = variety_params[variety]
        
        length_mm = np.random.normal(vp['l'], 0.2)
        width_mm = np.random.normal(vp['w'], 0.1)
        area_mm2 = np.random.normal(vp['a'], 0.5)
        aspect_ratio = length_mm / width_mm
        
        circularity = np.random.uniform(0.7, 0.9)
        solidity = np.random.uniform(0.9, 0.99)
        eccentricity = np.random.uniform(0.4, 0.8)
        
        rg_ratio = np.random.normal(1.1, 0.05)
        rb_ratio = np.random.normal(1.2, 0.05)
        gb_ratio = np.random.normal(1.05, 0.05)
        norm_s = np.random.uniform(0, 1)
        
        data.append({
            'variety': variety,
            'defect_type': defect,
            'length_mm': length_mm,
            'width_mm': width_mm,
            'aspect_ratio': aspect_ratio,
            'area_mm2': area_mm2,
            'circularity': circularity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'rg_ratio': rg_ratio,
            'rb_ratio': rb_ratio,
            'gb_ratio': gb_ratio,
            'norm_s': norm_s
        })
        
    return pd.DataFrame(data)

def load_data(filepath: str) -> pd.DataFrame:
    """Load real data or generate synthetic data if not found."""
    if os.path.exists(filepath):
        print(f"Loading real data from {filepath}...")
        return pd.read_csv(filepath)
    else:
        print(f"File {filepath} not found. Generating synthetic data...")
        return generate_synthetic_data()

def get_significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return 'ns'

def run_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Run One-way ANOVA for key features across varieties."""
    print("Running One-way ANOVA...")
    results = []
    
    for feature in KEY_FEATURES:
        groups = [group[feature].values for name, group in df.groupby('variety')]
        f_stat, p_val = stats.f_oneway(*groups)
        results.append({
            'Feature': feature,
            'F-statistic': f_stat,
            'p-value': p_val,
            'Significance': get_significance_marker(p_val)
        })
        
    return pd.DataFrame(results)

def run_kruskal_wallis(df: pd.DataFrame) -> List[Dict]:
    """Run Kruskal-Wallis H-test for key features across varieties."""
    print("Running Kruskal-Wallis test...")
    results = []
    
    for feature in KEY_FEATURES:
        groups = [group[feature].values for name, group in df.groupby('variety')]
        h_stat, p_val = stats.kruskal(*groups)
        results.append({
            'Feature': feature,
            'H-statistic': h_stat,
            'p-value': p_val,
            'Significance': get_significance_marker(p_val)
        })
        
    return results

def run_chi_square(df: pd.DataFrame) -> Tuple[float, float, int, pd.DataFrame]:
    """Run Chi-square test for independence between variety and defect type."""
    print("Running Chi-square test...")
    contingency = pd.crosstab(df['variety'], df['defect_type'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p_val, dof, contingency

def run_spearman_correlation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Run Spearman correlation between 11 features and return top 5 strongest."""
    print("Running Spearman correlation...")
    corr_matrix = df[FEATURES].corr(method='spearman')
    
    # Get top 5 strongest correlations (absolute value)
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0] # Remove self correlation
    
    # Drop duplicates like (A, B) and (B, A)
    corr_pairs = corr_pairs.loc[~corr_pairs.index.duplicated()]
    # Actually need to sort by value and handle duplicates in indices manually for pandas MultiIndex
    # Better way: get upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper = corr_matrix.abs().where(mask)
    top_5 = upper.unstack().dropna().sort_values(ascending=False).head(5)
    
    return corr_matrix, top_5

def run_tukey_hsd(df: pd.DataFrame) -> Tuple[Any, List[str]]:
    """Run Post-hoc Tukey HSD for length_mm across varieties."""
    print("Running Tukey HSD for length_mm...")
    
    groups = [group['length_mm'].values for name, group in df.groupby('variety')]
    variety_names = list(df.groupby('variety').groups.keys())
    
    try:
        # scipy.stats.tukey_hsd available in 3.10+
        res = stats.tukey_hsd(*groups)
        return res, variety_names
    except AttributeError:
        # Fallback if not available
        print("scipy.stats.tukey_hsd not available. Skipping Tukey test.")
        return None, variety_names

def generate_report(
    anova_res: pd.DataFrame, 
    kw_res: List[Dict], 
    chi2_res: Tuple[float, float, int], 
    top_corr: pd.Series, 
    tukey_res: Tuple[Any, List[str]],
    report_path: str
) -> None:
    """Generate and save formatted text report."""
    print(f"Generating report at {report_path}...")
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("STATISTICAL TESTING REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. ONE-WAY ANOVA\n")
        f.write("-" * 20 + "\n")
        for _, row in anova_res.iterrows():
            f.write(f"{row['Feature']}: F = {row['F-statistic']:.4f}, p = {row['p-value']:.4e} {row['Significance']}\n")
        f.write("\n")
        
        f.write("2. KRUSKAL-WALLIS TEST\n")
        f.write("-" * 20 + "\n")
        for res in kw_res:
            f.write(f"{res['Feature']}: H = {res['H-statistic']:.4f}, p = {res['p-value']:.4e} {res['Significance']}\n")
        f.write("\n")
        
        f.write("3. CHI-SQUARE TEST (Variety vs Defect Type)\n")
        f.write("-" * 20 + "\n")
        chi2, p_val, dof = chi2_res
        f.write(f"Chi2 = {chi2:.4f}, p = {p_val:.4e} {get_significance_marker(p_val)}, DOF = {dof}\n")
        f.write("\n")
        
        f.write("4. TOP 5 SPEARMAN CORRELATIONS\n")
        f.write("-" * 20 + "\n")
        for (f1, f2), val in top_corr.items():
            f.write(f"{f1} - {f2}: |r| = {val:.4f}\n")
        f.write("\n")
        
        f.write("5. TUKEY HSD POST-HOC (length_mm)\n")
        f.write("-" * 20 + "\n")
        tukey_obj, names = tukey_res
        if tukey_obj is not None:
            f.write("Results printed for pairwise variety comparisons:\n")
            f.write(str(tukey_obj))
        else:
            f.write("Tukey HSD test not performed.\n")
        f.write("\n")
        
        f.write("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    out_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    data_path = os.path.join(data_dir, 'grain_features.csv')
    df = load_data(data_path)
    
    # Run tests
    anova_df = run_anova(df)
    kw_results = run_kruskal_wallis(df)
    chi2, chi_p, chi_dof, _ = run_chi_square(df)
    corr_matrix, top_corr = run_spearman_correlation(df)
    tukey_res = run_tukey_hsd(df)
    
    # Save outputs
    anova_path = os.path.join(out_dir, 'anova_results.csv')
    corr_path = os.path.join(out_dir, 'correlation_matrix.csv')
    report_path = os.path.join(out_dir, 'statistical_report.txt')
    
    print(f"Saving ANOVA results to {anova_path}...")
    anova_df.to_csv(anova_path, index=False)
    
    print(f"Saving correlation matrix to {corr_path}...")
    corr_matrix.to_csv(corr_path)
    
    generate_report(anova_df, kw_results, (chi2, chi_p, chi_dof), top_corr, tukey_res, report_path)
    print("Statistical testing completed successfully.")

if __name__ == '__main__':
    main()
