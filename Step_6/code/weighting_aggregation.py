#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: Weighting and Aggregation for Community Risk Index

This script implements the weighting and aggregation step for the Community Risk Index,
following the OECD/JRC Handbook's guidelines on composite indicators.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import os

# Set figure style and parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directories if they don't exist
Path("Step_6/output/figures").mkdir(parents=True, exist_ok=True)

# Load normalized data from Step 5
normalized_data = pd.read_csv('Step_5/output/step5_normalized_dataset.csv')

# Define the pillars and their respective indicators
pillars = {
    'Demographics': ['racepctblack', 'racePctHisp', 'pctUrban', 'PctNotSpeakEnglWell'],
    'Income': ['medIncome', 'pctWPubAsst', 'PctPopUnderPov', 'PctUnemployed'],
    'Housing': ['PctFam2Par', 'PctHousOccup', 'PctVacantBoarded', 'PctHousNoPhone', 'PctSameHouse85'],
    'Crime': ['murdPerPop', 'robbbPerPop', 'autoTheftPerPop', 'arsonsPerPop', 'ViolentCrimesPerPop']
}

# Extract community names for later use
community_names = normalized_data['communityname'].copy()

# Function to calculate correlation matrix and create enhanced heatmap
def analyze_correlations(data, indicators, pillar_name):
    """
    Calculate correlations within a pillar and save enhanced heatmap visualization
    """
    corr_matrix = data[indicators].corr()
    
    # Check for highly correlated pairs (|r| > 0.9)
    high_corr_pairs = []
    for i in range(len(indicators)):
        for j in range(i+1, len(indicators)):
            var1 = indicators[i]
            var2 = indicators[j]
            corr_value = corr_matrix.loc[var1, var2]
            if abs(corr_value) > 0.9:
                high_corr_pairs.append((var1, var2, corr_value))
    
    # Create enhanced heatmap visualization
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Custom colormap for better contrast
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw heatmap with enhanced formatting
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                mask=mask, fmt='.2f', linewidths=0.5, square=True,
                cbar_kws={"shrink": .8}, annot_kws={"size": 10})
    
    plt.title(f'Correlation Matrix: {pillar_name} Pillar', fontsize=16, pad=20)
    
    # Add annotation for high correlations if any
    if high_corr_pairs:
        high_corr_text = "Highly correlated pairs (|r| > 0.9):"
        for pair in high_corr_pairs:
            high_corr_text += f"\n• {pair[0]} & {pair[1]}: r = {pair[2]:.2f}"
        plt.figtext(0.5, 0.01, high_corr_text, ha="center", fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(f'Step_6/output/figures/enhanced_corr_{pillar_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix, high_corr_pairs

# Function to apply PCA for weighting
def apply_pca_weights(data, indicators):
    """
    Apply PCA to derive weights based on explained variance
    """
    # Standardize the indicators first (PCA is sensitive to scale)
    scaler = lambda x: (x - x.mean()) / x.std()
    data_std = data[indicators].apply(scaler)
    
    # Fit PCA model
    pca = PCA(n_components=len(indicators))
    pca.fit(data_std)
    
    # Get explained variance ratios
    explained_var = pca.explained_variance_ratio_
    
    # Calculate weights based on explained variance
    weights = explained_var / explained_var.sum()
    
    return dict(zip(indicators, weights))

# Functions for different weighting schemes
def equal_weights(indicators):
    """Apply equal weights to all indicators"""
    n = len(indicators)
    return {ind: 1/n for ind in indicators}

def pca_weights(data, indicators):
    """Apply PCA-based weights"""
    return apply_pca_weights(data, indicators)

def stakeholder_weights(indicators, weights_dict):
    """Apply weights based on stakeholder input or expert opinion"""
    # Normalize weights to sum to 1
    total = sum(weights_dict.values())
    return {ind: weights_dict[ind]/total for ind in indicators}

# Create empty dictionaries to store weights and pillar scores
within_pillar_weights = {}
adjusted_weights = {}
pillar_scores = pd.DataFrame(index=normalized_data.index)

# Process each pillar
for pillar_name, indicators in pillars.items():
    print(f"\nProcessing {pillar_name} pillar...")
    
    # 1. Analyze correlations within the pillar
    corr_matrix, high_corr_pairs = analyze_correlations(normalized_data, indicators, pillar_name)
    
    # 2. Choose weighting method for this pillar
    # We'll use PCA weights for all pillars in this implementation, but this can be customized
    if pillar_name == 'Crime':
        # For Crime pillar, we use equal weights since all crime types are equally important
        weights = equal_weights(indicators)
        weight_method = "Equal Weights (EW)"
    else:
        # For other pillars, we use PCA weights
        weights = pca_weights(normalized_data, indicators)
        weight_method = "PCA-variance weights"
    
    print(f"  Using {weight_method} for {pillar_name} pillar")
    within_pillar_weights[pillar_name] = weights
    
    # 3. Adjust weights if needed to account for high correlation
    adjusted = weights.copy()
    if high_corr_pairs:
        print(f"  Found {len(high_corr_pairs)} highly correlated pairs:")
        for var1, var2, corr in high_corr_pairs:
            print(f"    {var1} & {var2}: r = {corr:.4f}")
            # Down-weight both variables
            factor = 0.5  # Reduce weight by half
            adjusted[var1] *= factor
            adjusted[var2] *= factor
            
        # Renormalize weights to sum to 1
        total = sum(adjusted.values())
        adjusted = {ind: w/total for ind, w in adjusted.items()}
        print("  Weights adjusted for high correlation")
    
    adjusted_weights[pillar_name] = adjusted
    
    # 4. Calculate pillar scores using adjusted weights
    # For each community, calculate weighted sum of indicators
    pillar_score = pd.Series(0, index=normalized_data.index)
    for indicator, weight in adjusted.items():
        pillar_score += normalized_data[indicator] * weight
    
    # Add pillar score to results DataFrame
    pillar_scores[f'{pillar_name}Score'] = pillar_score
    print(f"  {pillar_name} pillar scores calculated")

# Create a combined correlation matrix visualization
plt.figure(figsize=(20, 16))
for i, (pillar_name, indicators) in enumerate(pillars.items(), 1):
    plt.subplot(2, 2, i)
    corr_matrix = normalized_data[indicators].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                mask=mask, fmt='.2f', linewidths=0.5, annot_kws={"size": 8})
    plt.title(f'{pillar_name} Pillar')

plt.suptitle('Correlation Matrices by Pillar', fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Step_6/output/figures/all_correlation_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# Save within-pillar weights to CSV
pd.DataFrame(within_pillar_weights).to_csv('Step_6/output/weights_within.csv')
pd.DataFrame(adjusted_weights).to_csv('Step_6/output/weights_within_adjusted.csv')

# Add community names back to pillar scores
pillar_scores.insert(0, 'communityname', community_names)

# Save pillar scores to CSV
pillar_scores.to_csv('Step_6/output/pillar_scores.csv', index=False)

# Calculate between-pillar weights (cross-pillar)
pillar_columns = [col for col in pillar_scores.columns if col.endswith('Score')]
pillar_scores_only = pillar_scores[pillar_columns]

# Generate three weighting scenarios
# Scenario 1: Equal weights
equal_cross_weights = equal_weights(pillar_columns)
print("\nScenario 1: Equal weights between pillars")
print(equal_cross_weights)

# Scenario 2: PCA weights
pca_cross_weights = pca_weights(pillar_scores, pillar_columns)
print("\nScenario 2: PCA-variance weights between pillars")
print(pca_cross_weights)

# Scenario 3: Stakeholder weights
# These weights are hypothetical and would come from stakeholder input
stakeholder_dict = {
    'DemographicsScore': 0.20,  # Community stakeholders
    'IncomeScore': 0.30,        # Local government
    'HousingScore': 0.20,       # Housing authorities
    'CrimeScore': 0.30          # Law enforcement
}
stakeholder_cross_weights = stakeholder_weights(pillar_columns, stakeholder_dict)
print("\nScenario 3: Stakeholder weights between pillars")
print(stakeholder_cross_weights)

# Save all cross-pillar weighting scenarios
cross_weights_df = pd.DataFrame({
    'EqualWeights': equal_cross_weights,
    'PCAWeights': pca_cross_weights,
    'StakeholderWeights': stakeholder_cross_weights
})
cross_weights_df.to_csv('Step_6/output/weights_between.csv')

# Calculate final composite scores for each scenario
composite_scores = pillar_scores.copy()

# Scenario 1: Equal weights
composite_scores['CRI_EqualWeights'] = 0
for pillar, weight in equal_cross_weights.items():
    composite_scores['CRI_EqualWeights'] += composite_scores[pillar] * weight

# Scenario 2: PCA weights
composite_scores['CRI_PCAWeights'] = 0
for pillar, weight in pca_cross_weights.items():
    composite_scores['CRI_PCAWeights'] += composite_scores[pillar] * weight

# Scenario 3: Stakeholder weights
composite_scores['CRI_StakeholderWeights'] = 0
for pillar, weight in stakeholder_cross_weights.items():
    composite_scores['CRI_StakeholderWeights'] += composite_scores[pillar] * weight

# Save composite scores
composite_scores.to_csv('Step_6/output/composite_scores.csv', index=False)

# Create ranking for each scenario
for scenario in ['CRI_EqualWeights', 'CRI_PCAWeights', 'CRI_StakeholderWeights']:
    composite_scores[f'{scenario}_Rank'] = composite_scores[scenario].rank(ascending=False).astype(int)

# Save ranked scores
composite_scores.to_csv('Step_6/output/composite_scores_ranked.csv', index=False)

# Calculate rank differences for sensitivity analysis
composite_scores['RankDiff_Equal_PCA'] = abs(
    composite_scores['CRI_EqualWeights_Rank'] - composite_scores['CRI_PCAWeights_Rank']
)
composite_scores['RankDiff_Equal_Stakeholder'] = abs(
    composite_scores['CRI_EqualWeights_Rank'] - composite_scores['CRI_StakeholderWeights_Rank']
)

# Create visualizations
# 1. Bar chart for top 20 communities (ranked by equal weights)
top_20 = composite_scores.sort_values('CRI_EqualWeights', ascending=False).head(20)
plt.figure(figsize=(14, 10))
# Use a better colormap
bars = plt.barh(top_20['communityname'], top_20['CRI_EqualWeights'], 
                color=plt.colormaps['viridis'](np.linspace(0, 0.8, len(top_20))))

# Add value labels at the end of each bar
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f"{top_20['CRI_EqualWeights'].iloc[i]:.3f}", 
             va='center', fontweight='bold')

plt.title('Top 20 Communities by Crime-Risk Index (Equal Weights)', fontsize=16)
plt.xlabel('Crime-Risk Index Score (higher is better)', fontsize=14)
plt.ylabel('Community', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Step_6/output/figures/enhanced_top_20_communities.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Improved radar chart for pillar scores (top 5 and bottom 5 communities)
def create_improved_radar_chart(data, title, filename):
    # Prepare data for radar chart
    categories = ['Demographics', 'Income', 'Housing', 'Crime']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Use color palette
    colors = plt.colormaps['viridis']
    color_values = np.linspace(0, 0.8, len(data))
    
    # Add each community
    for i, (idx, row) in enumerate(data.iterrows()):
        community = row['communityname']
        values = [row['DemographicsScore'], row['IncomeScore'], 
                  row['HousingScore'], row['CrimeScore']]
        values += values[:1]  # Close the loop
        
        # Plot community with color
        color = colors(color_values[i])
        ax.plot(angles, values, linewidth=2, label=community, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=14)
    
    # Add grid lines
    plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], size=12)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Add title
    plt.title(title, size=18, y=1.08)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create improved radar charts
top_5 = composite_scores.sort_values('CRI_EqualWeights', ascending=False).head(5)
bottom_5 = composite_scores.sort_values('CRI_EqualWeights').head(5)

create_improved_radar_chart(top_5, 'Pillar Scores for Top 5 Communities', 
                           'Step_6/output/figures/improved_radar_top5.png')
create_improved_radar_chart(bottom_5, 'Pillar Scores for Bottom 5 Communities', 
                           'Step_6/output/figures/improved_radar_bottom5.png')

# Combined top-bottom radar chart
plt.figure(figsize=(18, 9))

# Top 5 on left
plt.subplot(1, 2, 1, polar=True)
categories = ['Demographics', 'Income', 'Housing', 'Crime']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

plt.xticks(angles[:-1], categories, size=14)
plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], size=10)
plt.ylim(0, 1)

# Plot each top community with distinct colors
colors = plt.colormaps['viridis']
color_values = np.linspace(0, 0.8, len(top_5))

for i, (idx, row) in enumerate(top_5.iterrows()):
    values = [row['DemographicsScore'], row['IncomeScore'], 
              row['HousingScore'], row['CrimeScore']]
    values += values[:1]  # Close the loop
    color = colors(color_values[i])
    plt.plot(angles, values, linewidth=2, label=row['communityname'], color=color)
    plt.fill(angles, values, alpha=0.1, color=color)

plt.title('Top 5 Communities', size=16)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

# Bottom 5 on right
plt.subplot(1, 2, 2, polar=True)
plt.xticks(angles[:-1], categories, size=14)
plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], size=10)
plt.ylim(0, 1)

# Plot each bottom community with distinct colors
colors = plt.colormaps['plasma']
color_values = np.linspace(0, 0.8, len(bottom_5))

for i, (idx, row) in enumerate(bottom_5.iterrows()):
    values = [row['DemographicsScore'], row['IncomeScore'], 
              row['HousingScore'], row['CrimeScore']]
    values += values[:1]  # Close the loop
    color = colors(color_values[i])
    plt.plot(angles, values, linewidth=2, label=row['communityname'], color=color)
    plt.fill(angles, values, alpha=0.1, color=color)

plt.title('Bottom 5 Communities', size=16)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

plt.suptitle('Comparison: Top 5 vs. Bottom 5 Communities', fontsize=20, y=0.98)
plt.tight_layout()
plt.savefig('Step_6/output/figures/top_vs_bottom_radar.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Improved sensitivity analysis - rank shifts between scenarios
# Get communities with largest rank shifts
top_rank_shifts = composite_scores.sort_values('RankDiff_Equal_PCA', ascending=False).head(15)

plt.figure(figsize=(14, 10))
plt.hlines(y=top_rank_shifts['communityname'], 
           xmin=top_rank_shifts['CRI_EqualWeights_Rank'], 
           xmax=top_rank_shifts['CRI_PCAWeights_Rank'], 
           color='blue', alpha=0.7, linewidth=2.5)

plt.scatter(top_rank_shifts['CRI_EqualWeights_Rank'], top_rank_shifts['communityname'], 
            color='navy', s=120, label='Equal Weights', zorder=10)
plt.scatter(top_rank_shifts['CRI_PCAWeights_Rank'], top_rank_shifts['communityname'], 
            color='darkorange', s=120, label='PCA Weights', zorder=10)

# Add rank difference annotations
for i, row in top_rank_shifts.iterrows():
    diff = row['RankDiff_Equal_PCA']
    midpoint = (row['CRI_EqualWeights_Rank'] + row['CRI_PCAWeights_Rank']) / 2
    plt.text(midpoint, row['communityname'], f" Δ{int(diff)}", 
             fontweight='bold', va='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

plt.title('Communities with Largest Rank Shifts: Equal vs. PCA Weights', fontsize=16)
plt.xlabel('Rank Position (lower is better)', fontsize=14)
plt.ylabel('Community', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Step_6/output/figures/improved_rank_shift_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Similar visualization for Equal vs. Stakeholder weights
top_stakeholder_shifts = composite_scores.sort_values('RankDiff_Equal_Stakeholder', ascending=False).head(15)

plt.figure(figsize=(14, 10))
plt.hlines(y=top_stakeholder_shifts['communityname'], 
           xmin=top_stakeholder_shifts['CRI_EqualWeights_Rank'], 
           xmax=top_stakeholder_shifts['CRI_StakeholderWeights_Rank'], 
           color='green', alpha=0.7, linewidth=2.5)

plt.scatter(top_stakeholder_shifts['CRI_EqualWeights_Rank'], top_stakeholder_shifts['communityname'], 
            color='navy', s=120, label='Equal Weights', zorder=10)
plt.scatter(top_stakeholder_shifts['CRI_StakeholderWeights_Rank'], top_stakeholder_shifts['communityname'], 
            color='crimson', s=120, label='Stakeholder Weights', zorder=10)

for i, row in top_stakeholder_shifts.iterrows():
    diff = row['RankDiff_Equal_Stakeholder']
    midpoint = (row['CRI_EqualWeights_Rank'] + row['CRI_StakeholderWeights_Rank']) / 2
    plt.text(midpoint, row['communityname'], f" Δ{int(diff)}", 
             fontweight='bold', va='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

plt.title('Communities with Largest Rank Shifts: Equal vs. Stakeholder Weights', fontsize=16)
plt.xlabel('Rank Position (lower is better)', fontsize=14)
plt.ylabel('Community', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Step_6/output/figures/improved_rank_shift_stakeholder.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Quality checks and summary statistics
# Check that all pillar scores are between 0 and 1
min_scores = pillar_scores_only.min().min()
max_scores = pillar_scores_only.max().max()
print(f"\nQuality check - Score range: min={min_scores:.4f}, max={max_scores:.4f}")

# Check that no composite scores are NaN
nan_count = composite_scores[['CRI_EqualWeights', 'CRI_PCAWeights', 'CRI_StakeholderWeights']].isna().sum().sum()
print(f"Quality check - NaN values in composite scores: {nan_count}")

# Summary statistics for composite scores
summary_stats = composite_scores[['CRI_EqualWeights', 'CRI_PCAWeights', 'CRI_StakeholderWeights']].describe()
summary_stats.to_csv('Step_6/output/composite_score_summary.csv')
print("\nSummary statistics for composite scores:")
print(summary_stats)

print("\nStep 6 completed successfully with enhanced visualizations!") 