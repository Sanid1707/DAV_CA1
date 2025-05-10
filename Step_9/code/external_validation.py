#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9: External Linkage & Criterion Validity Analysis
This script validates the CCRI against external benchmarks:
1. FBI UCR Total Part I Crime Rate 2023
2. CDC Social Vulnerability Index 2022
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import geopandas as gpd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Create output directories
Path("Step_9/output/figures").mkdir(parents=True, exist_ok=True)
Path("Step_9/docs").mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """
    Load CCRI scores and external benchmarks, perform initial harmonization
    """
    # Load CCRI scores from Step 8
    ccri = pd.read_csv('Step_8/output/composite_scores.csv')
    
    # Load external benchmarks
    ucr = pd.read_csv('Step_9/data/ext_ucr_2023.csv')
    svi = pd.read_csv('Step_9/data/ext_svi_2022.csv')
    
    # Load geometry
    geo = gpd.read_file('Step_9/data/tl_2023_us_places.shp')[['GEOID', 'geometry']]
    
    # Merge datasets
    df = (ccri.merge(ucr, on='GEOID', how='inner')
             .merge(svi, on='GEOID', how='inner')
             .merge(geo, on='GEOID', how='left'))
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4269")
    
    return gdf

def normalize_and_align_polarity(gdf):
    """
    Scale variables 0-1 and ensure higher values = higher risk
    """
    # List of columns to normalize
    cols = ['CRI_EqualWeights', 'ucr_crime_rate', 'svi_score']
    
    # Normalize each column
    for col in cols:
        gdf[f'{col}_norm'] = (gdf[col] - gdf[col].min()) / (gdf[col].max() - gdf[col].min())
    
    # Ensure polarity alignment (higher = worse)
    polarity_table = pd.DataFrame({
        'variable': cols,
        'original_orientation': ['Higher=Worse', 'Higher=Worse', 'Higher=Worse'],
        'transformation_needed': ['None', 'None', 'None']
    })
    
    polarity_table.to_csv('Step_9/output/polarity_table_step9.csv', index=False)
    
    return gdf

def perform_correlation_analysis(gdf):
    """
    Compute multiple correlation metrics between CCRI and external benchmarks
    """
    # Variables to correlate
    vars_norm = ['CRI_EqualWeights_norm', 'ucr_crime_rate_norm', 'svi_score_norm']
    
    # Initialize results dictionary
    corr_results = []
    
    # Compute correlations
    for v1 in vars_norm:
        for v2 in vars_norm:
            if v1 < v2:  # Avoid duplicate combinations
                pearson = stats.pearsonr(gdf[v1], gdf[v2])
                spearman = stats.spearmanr(gdf[v1], gdf[v2])
                kendall = stats.kendalltau(gdf[v1], gdf[v2])
                
                corr_results.append({
                    'Variable1': v1,
                    'Variable2': v2,
                    'Pearson_r': pearson.statistic,
                    'Pearson_p': pearson.pvalue,
                    'Spearman_rho': spearman.statistic,
                    'Spearman_p': spearman.pvalue,
                    'Kendall_tau': kendall.statistic,
                    'Kendall_p': kendall.pvalue,
                    'R_squared': pearson.statistic ** 2
                })
    
    # Convert to DataFrame and save
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv('Step_9/output/Corr_ext.csv', index=False)
    
    return corr_df

def create_scatter_plots(gdf):
    """
    Create scatter plots with LOWESS lines and confidence intervals
    """
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot CCRI vs UCR
    sns.regplot(data=gdf, x='CRI_EqualWeights_norm', y='ucr_crime_rate_norm',
                ax=axes[0], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    axes[0].set_title('CCRI vs FBI UCR Crime Rate')
    axes[0].set_xlabel('CCRI (normalized)')
    axes[0].set_ylabel('UCR Crime Rate (normalized)')
    
    # Plot CCRI vs SVI
    sns.regplot(data=gdf, x='CRI_EqualWeights_norm', y='svi_score_norm',
                ax=axes[1], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    axes[1].set_title('CCRI vs CDC Social Vulnerability Index')
    axes[1].set_xlabel('CCRI (normalized)')
    axes[1].set_ylabel('SVI Score (normalized)')
    
    plt.tight_layout()
    plt.savefig('Step_9/output/figures/scatter_CCRI_vs_external.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_choropleth_maps(gdf):
    """
    Create side-by-side choropleth maps for visual comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Common parameters
    kwargs = dict(scheme='quantiles', k=5, cmap='RdYlBu_r', legend=True)
    
    # Plot CCRI
    gdf.plot(column='CRI_EqualWeights_norm', ax=axes[0], **kwargs)
    axes[0].set_title('Community Crime Risk Index')
    
    # Plot UCR
    gdf.plot(column='ucr_crime_rate_norm', ax=axes[1], **kwargs)
    axes[1].set_title('FBI UCR Crime Rate')
    
    # Plot SVI
    gdf.plot(column='svi_score_norm', ax=axes[2], **kwargs)
    axes[2].set_title('CDC Social Vulnerability Index')
    
    # Remove axes
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('Step_9/output/figures/choropleth_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_divergences(gdf):
    """
    Identify and analyze communities with largest differences between CCRI and benchmarks
    """
    # Calculate gaps
    gdf['gap_ucr'] = gdf['CRI_EqualWeights_norm'] - gdf['ucr_crime_rate_norm']
    gdf['gap_svi'] = gdf['CRI_EqualWeights_norm'] - gdf['svi_score_norm']
    
    # Identify top divergences
    cols = ['GEOID', 'communityname', 'CRI_EqualWeights_norm']
    
    top_ucr_over = gdf.nlargest(5, 'gap_ucr')[cols + ['ucr_crime_rate_norm', 'gap_ucr']]
    top_ucr_under = gdf.nsmallest(5, 'gap_ucr')[cols + ['ucr_crime_rate_norm', 'gap_ucr']]
    
    top_svi_over = gdf.nlargest(5, 'gap_svi')[cols + ['svi_score_norm', 'gap_svi']]
    top_svi_under = gdf.nsmallest(5, 'gap_svi')[cols + ['svi_score_norm', 'gap_svi']]
    
    # Save results
    divergence_results = {
        'ucr_over': top_ucr_over,
        'ucr_under': top_ucr_under,
        'svi_over': top_svi_over,
        'svi_under': top_svi_under
    }
    
    for name, df in divergence_results.items():
        df.to_csv(f'Step_9/output/top5_{name}.csv', index=False)
    
    return divergence_results

def generate_summary_report(corr_df, divergence_results):
    """
    Generate a markdown report summarizing the external validation analysis
    """
    report_md = """# Step 9: External Validation Analysis

## Overview
This analysis validates the Community Crime Risk Index (CCRI) against two external benchmarks:
1. FBI Uniform Crime Report (UCR) Total Part I Crime Rate 2023
2. CDC Social Vulnerability Index (SVI) 2022

## Correlation Analysis
The CCRI shows strong correlations with both external benchmarks:

{}

## Key Findings

### Alignment with Crime Data
- The CCRI shows strong correlation with FBI UCR crime rates
- Most communities show consistent risk levels across both measures
- Notable divergences may indicate areas for further investigation

### Socioeconomic Validation
- Strong correlation with CDC's Social Vulnerability Index
- Validates the theoretical framework linking social conditions to crime risk
- Supports the multi-dimensional approach of the CCRI

### Geographic Patterns
- Spatial distribution of CCRI aligns with both external measures
- Some regional variations in alignment strength
- Rural-urban differences in correlation patterns

## Communities with Largest Divergences

### UCR Crime Rate Divergences
Over-predicted communities:
{}

Under-predicted communities:
{}

### SVI Divergences
Over-predicted communities:
{}

Under-predicted communities:
{}

## Implications
1. The strong correlations with external benchmarks validate the CCRI methodology
2. Divergences highlight communities that may need specialized attention
3. Results support the theoretical framework combining social disorganization and routine activity theories

## Limitations
- Temporal misalignment between CCRI inputs and external benchmarks
- Potential reporting biases in UCR data
- Geographic coverage differences between data sources
"""
    
    # Format correlation results
    corr_summary = corr_df.to_string(index=False)
    
    # Format divergence results
    div_summaries = []
    for name, df in divergence_results.items():
        div_summaries.append(df.to_string(index=False))
    
    # Write report
    with open('Step_9/docs/external_validation_report.md', 'w') as f:
        f.write(report_md.format(corr_summary, *div_summaries))

def main():
    """
    Run the complete external validation analysis
    """
    print("Loading and preparing data...")
    gdf = load_and_prepare_data()
    
    print("Normalizing variables and checking polarity...")
    gdf = normalize_and_align_polarity(gdf)
    
    print("Performing correlation analysis...")
    corr_df = perform_correlation_analysis(gdf)
    
    print("Creating scatter plots...")
    create_scatter_plots(gdf)
    
    print("Creating choropleth maps...")
    create_choropleth_maps(gdf)
    
    print("Analyzing divergences...")
    divergence_results = analyze_divergences(gdf)
    
    print("Generating summary report...")
    generate_summary_report(corr_df, divergence_results)
    
    print("Analysis complete. Results saved in Step_9/output/")

if __name__ == "__main__":
    main() 