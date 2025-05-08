#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Multivariate Analysis - Sections G-I
- Create decision grid for indicator selection
- Re-run PCA on trimmed indicator set
- Document results and prepare for Step 5
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from collections import Counter
import matplotlib.ticker as mtick
import scipy.stats as stats
import warnings
import re

# Create directories if they don't exist
os.makedirs('Step_4/output/figures', exist_ok=True)

print("\n" + "="*80)
print("SECTIONS 4-G to 4-I: INDICATOR SELECTION, PCA RERUN, AND DOCUMENTATION")
print("="*80)

# Load data
try:
    # Load the original data
    df = pd.read_csv('step3_final_dataset_for_multivariate.csv')
    print(f"Loaded original data with {df.shape[0]} communities and {df.shape[1]} variables")
    
    # Load indicator decision grid
    indicator_decisions = pd.read_csv('Step_4/output/indicator_decision_grid.csv')
    print(f"Loaded indicator decision grid with {indicator_decisions.shape[0]} indicators")
    
except FileNotFoundError:
    try:
        # Alternative paths
        df = pd.read_csv('step3_imputed_dataset.csv')
        indicator_decisions = pd.read_csv('indicator_decision_grid.csv')
        print(f"Loaded data from alternative paths")
    except:
        print("Error: Could not load required data files")
        print("Please run Step_4_collinearity.py first to generate the indicator decision grid")
        exit(1)

# Define pillars and their indicators
pillars = {
    'Demographics': [
        'population', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 
        'pctUrban', 'PctImmigRec5', 'PctRecentImmig', 'PctNotSpeakEnglWell'
    ],
    'Income': [
        'medIncome', 'pctWSocSec', 'pctWPubAsst', 'PctPopUnderPov', 'PctUnemployed'
    ],
    'Housing': [
        'PersPerFam', 'PctFam2Par', 'PctLargHouseFam', 'PersPerRentOccHous', 
        'PctPersDenseHous', 'PctHousOccup', 'PctVacantBoarded', 'PctHousNoPhone',
        'PctWOFullPlumb', 'OwnOccQrange', 'RentQrange', 'NumInShelters', 'PctSameHouse85',
        'PctUsePubTrans'
    ],
    'Crime': [
        'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop',
        'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 
        'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
        'ViolentCrimesPerPop', 'nonViolPerPop'
    ]
}

# 4-G: Final decision grid for indicators
print("\n4-G: Creating final decision grid for indicators...")

# Create the final indicator selection
final_indicators = []

# Define minimum number of indicators to keep per pillar
MIN_INDICATORS_PER_PILLAR = 2

# Function to calculate indicator importance score
def calculate_importance_score(row, pillar_decisions):
    """
    Calculate a balanced importance score based on multiple criteria
    Returns a score between 0 and 1, where higher is more important
    """
    # Statistical quality (communality)
    communality = row['Communality'] if 'Communality' in row else 0.5
    
    # Uniqueness (inverse of being in high correlation group)
    uniqueness = 0.0 if row['In_High_Correlation_Group'] else 1.0
    
    # PC Representation - check if it's the top loading on any PC
    is_top_loading = False
    if 'PCA Top Loading' in row:
        is_top_loading = row['PCA Top Loading']
    else:
        # Try to infer from rationale or decision
        is_top_loading = 'representative' in row['Decision'].lower() or 'highest communality' in row['Rationale'].lower()
    
    pc_representation = 1.0 if is_top_loading else 0.5
    
    # Conceptual importance (default high for now - can be refined)
    conceptual_importance = 0.8
    
    # Calculate weighted score
    score = (
        0.3 * communality +         # Statistical quality
        0.2 * uniqueness +          # Uniqueness of information
        0.2 * pc_representation +   # Representation of principal components
        0.3 * conceptual_importance # Conceptual importance
    )
    
    return score

for pillar, variables in pillars.items():
    print(f"\nFinalizing indicators for {pillar} pillar...")
    
    # Get decisions for this pillar
    pillar_decisions = indicator_decisions[indicator_decisions['Pillar'] == pillar]
    
    if pillar_decisions.empty:
        print(f"  No decisions found for {pillar}. Keeping all variables.")
        for var in variables:
            if var in df.columns:
                final_indicators.append({
                    'Pillar': pillar,
                    'Indicator': var,
                    'Status': 'Keep',
                    'Rationale': 'No multivariate analysis available, keeping by default',
                    'Weight': 1.0  # Equal weights for now, to be adjusted in Step 6
                })
        continue
    
    # Initial classification based on decision
    keep_indicators = []
    consider_indicators = []
    drop_indicators = []
    
    # Calculate importance scores for all indicators
    importance_scores = {}
    for _, row in pillar_decisions.iterrows():
        indicator = row['Indicator']
        importance_scores[indicator] = calculate_importance_score(row, pillar_decisions)
        
        if row['Decision'].startswith('Keep'):
            keep_indicators.append(indicator)
        elif 'merging' in row['Decision']:
            consider_indicators.append(indicator)
        elif row['Decision'].startswith('Drop'):
            drop_indicators.append(indicator)
    
    # Sort indicators by importance score
    sorted_indicators = sorted(
        [(ind, importance_scores[ind]) for ind in variables if ind in importance_scores],
        key=lambda x: x[1],
        reverse=True  # Higher scores first
    )
    
    # Ensure minimum representation from each pillar
    final_keep_indicators = []
    
    # First, include all definite keeps
    final_keep_indicators.extend(keep_indicators)
    
    # If we don't have enough, add from consider_indicators by importance score
    if len(final_keep_indicators) < MIN_INDICATORS_PER_PILLAR:
        sorted_consider = sorted(
            [(ind, importance_scores[ind]) for ind in consider_indicators],
            key=lambda x: x[1],
            reverse=True
        )
        
        for ind, _ in sorted_consider:
            if ind not in final_keep_indicators:
                final_keep_indicators.append(ind)
                if len(final_keep_indicators) >= MIN_INDICATORS_PER_PILLAR:
                    break
    
    # If still not enough, add highest scoring dropped indicators
    if len(final_keep_indicators) < MIN_INDICATORS_PER_PILLAR:
        sorted_drop = sorted(
            [(ind, importance_scores[ind]) for ind in drop_indicators],
            key=lambda x: x[1],
            reverse=True
        )
        
        for ind, _ in sorted_drop:
            if ind not in final_keep_indicators:
                final_keep_indicators.append(ind)
                if len(final_keep_indicators) >= MIN_INDICATORS_PER_PILLAR:
                    break
    
    # For Crime and Income pillars, ensure PC1 is well-represented 
    # by keeping at least one high-loading indicator from PC1
    if pillar in ['Crime', 'Income']:
        has_pc1_representation = False
        for ind in final_keep_indicators:
            row = pillar_decisions[pillar_decisions['Indicator'] == ind]
            if not row.empty and ('PC1' in row['Rationale'].values[0] or 'principal component' in row['Rationale'].values[0].lower()):
                has_pc1_representation = True
                break
        
        if not has_pc1_representation:
            # Find an indicator that loads highly on PC1
            for ind, score in sorted_indicators:
                row = pillar_decisions[pillar_decisions['Indicator'] == ind]
                if not row.empty and ('PC1' in row['Rationale'].values[0] or 'principal component' in row['Rationale'].values[0].lower()):
                    if ind not in final_keep_indicators:
                        final_keep_indicators.append(ind)
                        break
    
    # Count final decisions
    keep_count = len(final_keep_indicators)
    drop_count = len([ind for ind in variables if ind in importance_scores and ind not in final_keep_indicators])
    
    print(f"  Balanced selection: Keep {keep_count}, Drop {drop_count}")
    print(f"  Indicators to keep: {', '.join(final_keep_indicators)}")
    
    # Add final decisions to the indicator list
    for _, row in pillar_decisions.iterrows():
        indicator = row['Indicator']
        decision = row['Decision']
        rationale = row['Rationale']
        
        # Determine final status based on our selection
        if indicator in final_keep_indicators:
            status = 'Keep'
            weight = 1.0
            
            # If it was originally marked for dropping, add explanation
            if decision.startswith('Drop'):
                rationale += ' Kept to ensure minimum pillar representation and balanced selection.'
            elif 'merging' in decision:
                rationale += ' Kept based on balanced scoring across statistical and conceptual criteria.'
        else:
            status = 'Drop'
            weight = 0.0
            
            # If it was originally marked for keeping, add explanation
            if decision.startswith('Keep'):
                rationale += ' Dropped due to sufficient representation from other indicators in this pillar.'
        
        # Add indicator with updated status to final list
        final_indicators.append({
            'Pillar': pillar,
            'Indicator': indicator,
            'Importance_Score': importance_scores.get(indicator, 0.0),
            'Status': status,
            'Rationale': rationale,
            'Weight': weight
        })

# Create final indicators dataframe
final_indicators_df = pd.DataFrame(final_indicators)
final_indicators_df.to_csv('Step_4/output/final_indicators.csv', index=False)
print(f"\nFinal indicators saved to 'Step_4/output/final_indicators.csv'")

# Create a pivoted view for easy reference
pivot_df = pd.pivot_table(
    final_indicators_df,
    index=['Pillar', 'Indicator'],
    values=['Status', 'Weight', 'Importance_Score'],
    aggfunc='first'
).reset_index()

pivot_df.to_csv('Step_4/output/final_indicators_pivot.csv', index=False)
print(f"Pivoted view saved to 'Step_4/output/final_indicators_pivot.csv'")

# Create final indicators summary
summary_stats = final_indicators_df.groupby(['Pillar', 'Status']).size().unstack(fill_value=0)
summary_stats['Total'] = summary_stats.sum(axis=1)
summary_stats.to_csv('Step_4/output/final_indicators_summary.csv')
print(f"Summary statistics saved to 'Step_4/output/final_indicators_summary.csv'")

# Print summary
print("\nFinal indicator selection summary:")
print(summary_stats)

# 4-H: Re-run PCA on trimmed indicator set
print("\n4-H: Re-running PCA on trimmed indicator set...")

# Function to standardize data
def standardize_data(df, variables):
    """Standardize variables in the dataframe"""
    result = df.copy()
    for var in variables:
        if var in df.columns:
            mean = df[var].mean()
            std = df[var].std()
            if std > 0:
                result[var] = (df[var] - mean) / std
    return result

# Identify indicators to keep for each pillar
kept_indicators = {}
for pillar in pillars.keys():
    pillar_indicators = final_indicators_df[
        (final_indicators_df['Pillar'] == pillar) & 
        (final_indicators_df['Status'].str.contains('Keep'))
    ]['Indicator'].tolist()
    
    if len(pillar_indicators) < 2:
        print(f"  Warning: Fewer than 2 indicators selected for {pillar}. PCA requires at least 2 variables.")
        # Keep all if too few selected
        pillar_indicators = [var for var in pillars[pillar] if var in df.columns]
        print(f"  Using all {len(pillar_indicators)} available indicators for {pillar}")
    
    kept_indicators[pillar] = pillar_indicators

# Perform PCA on the trimmed indicator set
pca_results_trimmed = {}

for pillar, variables in kept_indicators.items():
    print(f"\nRe-running PCA for {pillar} pillar with {len(variables)} indicators...")
    
    if len(variables) < 2:
        print(f"  Skipping PCA for {pillar} - insufficient variables")
        continue
    
    # Standardize data
    X = standardize_data(df, variables)[variables].values
    
    # Run PCA
    pca = PCA()
    pca.fit(X)
    
    # Extract key results
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Store results
    pca_results_trimmed[pillar] = {
        'components': pca.components_,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'variables': variables
    }
    
    # Summary statistics
    total_variance = sum(explained_variance)
    n_components_60pct = sum(cumulative_variance < 0.6) + 1
    
    print(f"  Total variance explained: {total_variance:.4f}")
    print(f"  Components needed for 60% variance: {n_components_60pct}")
    
    # Create scree plot for trimmed indicators
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), pca.explained_variance_, alpha=0.6)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title(f'Scree Plot - {pillar} (Trimmed)')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(1, len(explained_variance) + 1))
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'o-')
    plt.axhline(y=0.6, color='r', linestyle='--')
    plt.title(f'Cumulative Variance - {pillar} (Trimmed)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, len(explained_variance) + 1))
    
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/pca_trimmed_{pillar}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create component loadings visualization
    if len(variables) > 1:
        plt.figure(figsize=(12, 8))
        loadings = pd.DataFrame(
            pca.components_.T,
            index=variables,
            columns=[f'PC{i+1}' for i in range(len(variables))]
        )
        
        # Plot loadings for first 2 components
        components_to_plot = min(2, len(variables))
        sns.heatmap(
            loadings.iloc[:, :components_to_plot], 
            cmap='coolwarm', 
            center=0, 
            annot=True, 
            fmt='.2f'
        )
        plt.title(f'Component Loadings - {pillar} (Trimmed)')
        plt.tight_layout()
        plt.savefig(f'Step_4/output/figures/pca_loadings_trimmed_{pillar}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Compare results before and after trimming
pca_comparison = []

for pillar, results in pca_results_trimmed.items():
    # Calculate metrics
    variance_explained_pc1 = results['explained_variance'][0] if len(results['explained_variance']) > 0 else 0
    variance_explained_total = np.sum(results['explained_variance'])
    n_indicators = len(results['variables'])
    
    # Add to comparison
    pca_comparison.append({
        'Pillar': pillar,
        'Original_Indicators': len(pillars[pillar]),
        'Trimmed_Indicators': n_indicators,
        'Variance_Explained_PC1': variance_explained_pc1,
        'Total_Variance_Explained': variance_explained_total,
        'Interpretability': 'Improved' if variance_explained_pc1 > 0.4 else 'Similar'
    })

# Create comparison dataframe
pca_comparison_df = pd.DataFrame(pca_comparison)
pca_comparison_df.to_csv('Step_4/output/pca_comparison.csv', index=False)
print(f"\nPCA comparison saved to 'Step_4/output/pca_comparison.csv'")

# 4-I: Document and version-control
print("\n4-I: Documenting analysis and creating final dataset...")

# Create a markdown summary of the entire analysis
analysis_summary = """
# Step 4: Multivariate Analysis Summary

## Overview

This document summarizes the multivariate analysis conducted in Step 4, including:
- Principal Component Analysis (PCA) of indicators within each pillar
- Cluster analysis of communities
- Indicator collinearity assessment
- Final indicator selection

## Key Findings

### Principal Component Analysis

"""

# Add PCA findings for each pillar
for pillar, results in pca_results_trimmed.items():
    analysis_summary += f"#### {pillar} Pillar\n\n"
    analysis_summary += f"- Original indicators: {len(pillars[pillar])}\n"
    analysis_summary += f"- Final indicators: {len(results['variables'])}\n"
    
    if len(results['explained_variance']) > 0:
        variance_pc1 = results['explained_variance'][0]
        variance_total = np.sum(results['explained_variance'])
        analysis_summary += f"- Variance explained by PC1: {variance_pc1:.2f} ({variance_pc1*100:.1f}%)\n"
        analysis_summary += f"- Total variance explained: {variance_total:.2f} ({variance_total*100:.1f}%)\n"
    
    analysis_summary += f"- Final indicators: {', '.join(results['variables'])}\n\n"

analysis_summary += """
### Cluster Analysis

Cluster analysis was performed using both hierarchical and k-means clustering methods on the principal component scores. The optimal number of clusters was determined using silhouette scores.

"""

# Add cluster analysis findings
try:
    cluster_summary = pd.read_csv('Step_4/output/clustering_summary.csv')
    for _, row in cluster_summary.iterrows():
        pillar = row['Pillar']
        n_clusters = row['Optimal_Clusters']
        silhouette = row['Best_Silhouette_Score']
        rand_index = row['Rand_Index']
        
        analysis_summary += f"#### {pillar} Pillar\n\n"
        analysis_summary += f"- Optimal number of clusters: {n_clusters}\n"
        analysis_summary += f"- Silhouette score: {silhouette:.3f}\n"
        analysis_summary += f"- Agreement between methods (Rand Index): {rand_index:.3f}\n"
        analysis_summary += f"- See cluster profiles in '{pillar}_cluster_profiles.csv'\n\n"
except:
    analysis_summary += "Detailed cluster analysis information not available.\n\n"

analysis_summary += """
### Indicator Selection

The following summarizes the final indicator selection based on multivariate analysis:

"""

# Add indicator selection summary
for pillar, indicators in kept_indicators.items():
    analysis_summary += f"#### {pillar} Pillar\n\n"
    analysis_summary += f"- Selected indicators: {len(indicators)}\n"
    analysis_summary += f"- Indicators: {', '.join(indicators)}\n"
    
    # Add notes on dropped indicators
    dropped = [var for var in pillars[pillar] if var in df.columns and var not in indicators]
    if dropped:
        analysis_summary += f"- Dropped indicators: {', '.join(dropped)}\n"
    
    analysis_summary += "\n"

analysis_summary += """
### Implications for Next Steps

The final set of indicators identified through multivariate analysis will be:
1. Re-normalized in Step 5
2. Weighted and aggregated in Step 6

This ensures that the composite indicators will be built on statistically sound foundations with:
- Reduced redundancy between indicators
- Balanced representation across conceptual dimensions
- Enhanced interpretability of results
"""

# Save analysis summary
with open('Step_4/output/step4_analysis_summary.md', 'w') as f:
    f.write(analysis_summary)

print(f"Analysis summary saved to 'Step_4/output/step4_analysis_summary.md'")

# Create final dataset for next step
final_df = df.copy()

# Add an indicator column to mark variables that should be kept
for pillar, indicators in kept_indicators.items():
    for var in pillars[pillar]:
        if var in df.columns:
            col_name = f'keep_{var}'
            final_df[col_name] = var in indicators

# Save final dataset
final_df.to_csv('Step_4/output/step4_final_dataset.csv', index=False)
print(f"Final dataset saved to 'Step_4/output/step4_final_dataset.csv'")

# Create a simpler trimmed version with just the selected indicators
# Ensure communityname is preserved if it exists
columns_to_keep = ['communityname'] if 'communityname' in df.columns else []

# Add all kept indicators
for pillar, indicators in kept_indicators.items():
    columns_to_keep.extend([var for var in indicators if var in df.columns])

# Create and save the trimmed dataset
if len(columns_to_keep) > 0:
    trimmed_df = df[columns_to_keep].copy()
    trimmed_df.to_csv('Step_4/output/step4_trimmed_dataset.csv', index=False)
    print(f"Trimmed dataset with {len(columns_to_keep)} columns saved to 'Step_4/output/step4_trimmed_dataset.csv'")

print("\nStep 4 completed successfully!")
print("The data is now ready for normalization in Step 5.") 