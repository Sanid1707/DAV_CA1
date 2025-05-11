#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Visualizations for Step 6: Weighting and Aggregation

This script generates improved visualizations for the Community Risk Index,
focusing on correlation matrices, radar charts, and sensitivity analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set figure style and parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Ensure output directory exists
Path("Step_6/output/figures").mkdir(parents=True, exist_ok=True)

# Load required data
normalized_data = pd.read_csv('Step_5/output/step5_normalized_dataset.csv')
pillar_scores = pd.read_csv('Step_6/output/pillar_scores.csv')
composite_scores = pd.read_csv('Step_6/output/composite_scores_ranked.csv')

# Calculate rank differences if they don't exist
if 'RankDiff_Equal_PCA' not in composite_scores.columns:
    composite_scores['RankDiff_Equal_PCA'] = abs(
        composite_scores['CRI_EqualWeights_Rank'] - composite_scores['CRI_PCAWeights_Rank']
    )
    
if 'RankDiff_Equal_Stakeholder' not in composite_scores.columns:
    composite_scores['RankDiff_Equal_Stakeholder'] = abs(
        composite_scores['CRI_EqualWeights_Rank'] - composite_scores['CRI_StakeholderWeights_Rank']
    )

# Define pillars and their indicators
pillars = {
    'Demographics': ['racepctblack', 'racePctHisp', 'pctUrban', 'PctNotSpeakEnglWell'],
    'Income': ['medIncome', 'pctWPubAsst', 'PctPopUnderPov', 'PctUnemployed'],
    'Housing': ['PctFam2Par', 'PctHousOccup', 'PctVacantBoarded', 'PctHousNoPhone', 'PctSameHouse85'],
    'Crime': ['murdPerPop', 'robbbPerPop', 'autoTheftPerPop', 'arsonsPerPop', 'ViolentCrimesPerPop']
}

# 1. Enhanced Correlation Matrices
# Function to create an enhanced correlation heatmap
def create_enhanced_correlation_matrix(data, indicators, pillar_name):
    """
    Create an enhanced correlation matrix visualization with annotations
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
    
    # Create enhanced heatmap
    plt.figure(figsize=(12, 10))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                mask=mask, fmt='.2f', square=True, linewidths=0.5, 
                cbar_kws={"shrink": .8}, annot_kws={"size": 10})
    
    # Add title and annotations
    plt.title(f'Correlation Matrix: {pillar_name} Pillar', fontsize=18, pad=20)
    
    # Add text about highly correlated pairs if any
    if high_corr_pairs:
        high_corr_text = f"Highly correlated pairs (|r| > 0.9): "
        for pair in high_corr_pairs:
            high_corr_text += f"\n• {pair[0]} & {pair[1]}: r = {pair[2]:.2f}"
        plt.figtext(0.5, 0.01, high_corr_text, ha="center", fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(f'Step_6/output/figures/enhanced_corr_{pillar_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix, high_corr_pairs

# Create enhanced correlation matrices for each pillar
for pillar_name, indicators in pillars.items():
    print(f"Creating enhanced correlation matrix for {pillar_name} pillar...")
    create_enhanced_correlation_matrix(normalized_data, indicators, pillar_name)

# 2. Combined Correlation Visualization
# Create a figure with all four correlation matrices
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

# 3. Enhanced Radar Charts
def create_enhanced_radar_chart(data, title, filename, palette='tab10'):
    # Prepare data for radar chart
    categories = ['Demographics', 'Income', 'Housing', 'Crime']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Use a color palette (updated to avoid deprecation warning)
    colors = plt.colormaps[palette].resampled(len(data))
    
    # Add each community
    for i, (idx, row) in enumerate(data.iterrows()):
        community = row['communityname']
        values = [row['DemographicsScore'], row['IncomeScore'], 
                  row['HousingScore'], row['CrimeScore']]
        values += values[:1]  # Close the loop
        
        # Plot community with distinct color
        ax.plot(angles, values, linewidth=2, label=community, color=colors(i))
        ax.fill(angles, values, alpha=0.1, color=colors(i))
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=14)
    
    # Draw ylabels (0-1 scale)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
    ax.set_ylim(0, 1)
    
    # Add legend with better placement
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.title(title, size=18, y=1.08)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create enhanced radar charts
top_5 = composite_scores.sort_values('CRI_EqualWeights', ascending=False).head(5)
bottom_5 = composite_scores.sort_values('CRI_EqualWeights').head(5)

create_enhanced_radar_chart(top_5, 'Pillar Scores for Top 5 Communities', 
                           'Step_6/output/figures/enhanced_radar_top5.png', 'viridis')
create_enhanced_radar_chart(bottom_5, 'Pillar Scores for Bottom 5 Communities', 
                           'Step_6/output/figures/enhanced_radar_bottom5.png', 'plasma')

# 4. Combined Top-Bottom Radar Chart
# Create a figure with both top and bottom communities for comparison
plt.figure(figsize=(15, 7))

# Left subplot for top 5
plt.subplot(1, 2, 1, polar=True)
categories = ['Demographics', 'Income', 'Housing', 'Crime']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop
plt.xticks(angles[:-1], categories)
plt.ylim(0, 1)

for i, (idx, row) in enumerate(top_5.iterrows()):
    values = [row['DemographicsScore'], row['IncomeScore'], 
              row['HousingScore'], row['CrimeScore']]
    values += values[:1]  # Close the loop
    plt.plot(angles, values, linewidth=2, label=row['communityname'])
    plt.fill(angles, values, alpha=0.1)

plt.title('Top 5 Communities', size=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Right subplot for bottom 5
plt.subplot(1, 2, 2, polar=True)
plt.xticks(angles[:-1], categories)
plt.ylim(0, 1)

for i, (idx, row) in enumerate(bottom_5.iterrows()):
    values = [row['DemographicsScore'], row['IncomeScore'], 
              row['HousingScore'], row['CrimeScore']]
    values += values[:1]  # Close the loop
    plt.plot(angles, values, linewidth=2, label=row['communityname'])
    plt.fill(angles, values, alpha=0.1)

plt.title('Bottom 5 Communities', size=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.suptitle('Comparison of Top 5 vs Bottom 5 Communities', fontsize=18, y=1.05)
plt.tight_layout()
plt.savefig('Step_6/output/figures/top_vs_bottom_radar.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Enhanced Rank Shifts Plot
# Get communities with largest rank shifts for both comparisons
top_rank_shifts_pca = composite_scores.sort_values('RankDiff_Equal_PCA', ascending=False).head(10)
top_rank_shifts_stakeholder = composite_scores.sort_values('RankDiff_Equal_Stakeholder', ascending=False).head(10)

# Enhanced PCA rank shift plot
plt.figure(figsize=(14, 10))
# Plot horizontal lines connecting ranks
plt.hlines(y=top_rank_shifts_pca['communityname'], 
           xmin=top_rank_shifts_pca['CRI_EqualWeights_Rank'], 
           xmax=top_rank_shifts_pca['CRI_PCAWeights_Rank'], 
           color='blue', alpha=0.7, linewidth=2.5)

# Plot points for equal weights
plt.scatter(top_rank_shifts_pca['CRI_EqualWeights_Rank'], top_rank_shifts_pca['communityname'], 
            color='navy', s=120, label='Equal Weights', zorder=10)

# Plot points for PCA weights
plt.scatter(top_rank_shifts_pca['CRI_PCAWeights_Rank'], top_rank_shifts_pca['communityname'], 
            color='darkorange', s=120, label='PCA Weights', zorder=10)

# Add rank shift values as annotations
for i, row in top_rank_shifts_pca.iterrows():
    diff = row['RankDiff_Equal_PCA']
    midpoint = (row['CRI_EqualWeights_Rank'] + row['CRI_PCAWeights_Rank']) / 2
    plt.text(midpoint, row['communityname'], f" Δ{diff}", 
             fontweight='bold', va='center', ha='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

plt.title('Communities with Largest Rank Shifts: Equal vs. PCA Weights', fontsize=16)
plt.xlabel('Rank Position (lower is better)', fontsize=14)
plt.ylabel('Community', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Step_6/output/figures/enhanced_rank_shift_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# Enhanced Stakeholder rank shift plot
plt.figure(figsize=(14, 10))
plt.hlines(y=top_rank_shifts_stakeholder['communityname'], 
           xmin=top_rank_shifts_stakeholder['CRI_EqualWeights_Rank'], 
           xmax=top_rank_shifts_stakeholder['CRI_StakeholderWeights_Rank'], 
           color='green', alpha=0.7, linewidth=2.5)

plt.scatter(top_rank_shifts_stakeholder['CRI_EqualWeights_Rank'], top_rank_shifts_stakeholder['communityname'], 
            color='navy', s=120, label='Equal Weights', zorder=10)
plt.scatter(top_rank_shifts_stakeholder['CRI_StakeholderWeights_Rank'], top_rank_shifts_stakeholder['communityname'], 
            color='crimson', s=120, label='Stakeholder Weights', zorder=10)

for i, row in top_rank_shifts_stakeholder.iterrows():
    diff = row['RankDiff_Equal_Stakeholder']
    midpoint = (row['CRI_EqualWeights_Rank'] + row['CRI_StakeholderWeights_Rank']) / 2
    plt.text(midpoint, row['communityname'], f" Δ{diff}", 
             fontweight='bold', va='center', ha='center',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

plt.title('Communities with Largest Rank Shifts: Equal vs. Stakeholder Weights', fontsize=16)
plt.xlabel('Rank Position (lower is better)', fontsize=14)
plt.ylabel('Community', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('Step_6/output/figures/enhanced_rank_shift_stakeholder.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Enhanced Bar Chart for Top Communities
# Sort by CRI score and get top 15
top_15 = composite_scores.sort_values('CRI_EqualWeights', ascending=False).head(15)

plt.figure(figsize=(14, 10))
bars = plt.barh(top_15['communityname'], top_15['CRI_EqualWeights'], 
                color=plt.colormaps['viridis'](np.linspace(0, 0.8, len(top_15))))

# Add value labels at the end of each bar
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f"{top_15['CRI_EqualWeights'].iloc[i]:.3f}", 
             va='center', fontweight='bold')

plt.title('Top 15 Communities by Crime-Risk Index (Equal Weights)', fontsize=16)
plt.xlabel('Crime-Risk Index Score (higher is better)', fontsize=14)
plt.ylabel('Community', fontsize=14)
plt.xlim(0, max(top_15['CRI_EqualWeights']) * 1.15)  # Add some space for labels
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Step_6/output/figures/enhanced_top_15_communities.png', dpi=300, bbox_inches='tight')
plt.close()

print("Enhanced visualizations created successfully!") 