#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Radar Charts for Step 6

This script generates enhanced radar charts for top/bottom communities
and rank shift visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set figure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory if it doesn't exist
Path("Step_6/output/figures").mkdir(parents=True, exist_ok=True)

# Load required data
pillar_scores = pd.read_csv('Step_6/output/pillar_scores.csv')
composite_scores = pd.read_csv('Step_6/output/composite_scores_ranked.csv')

# Calculate rank differences
composite_scores['RankDiff_Equal_PCA'] = abs(
    composite_scores['CRI_EqualWeights_Rank'] - composite_scores['CRI_PCAWeights_Rank']
)
composite_scores['RankDiff_Equal_Stakeholder'] = abs(
    composite_scores['CRI_EqualWeights_Rank'] - composite_scores['CRI_StakeholderWeights_Rank']
)

print("Creating enhanced radar charts...")

# Function to create improved radar charts
def create_improved_radar_chart(data, title, filename):
    """Create an improved radar chart for the given communities."""
    # Prepare data for radar chart
    categories = ['Demographics', 'Income', 'Housing', 'Crime']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # Create a color palette (using numpy instead of matplotlib cmap directly)
    colors = plt.colormaps['viridis']
    num_colors = len(data)
    color_values = np.linspace(0, 0.8, num_colors)
    
    # Add each community
    for i, (idx, row) in enumerate(data.iterrows()):
        community = row['communityname']
        values = [row['DemographicsScore'], row['IncomeScore'], 
                  row['HousingScore'], row['CrimeScore']]
        values += values[:1]  # Close the loop
        
        # Plot community with color from the palette
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

# Get top and bottom 5 communities
top_5 = composite_scores.sort_values('CRI_EqualWeights', ascending=False).head(5)
bottom_5 = composite_scores.sort_values('CRI_EqualWeights').head(5)

# Create radar charts
create_improved_radar_chart(top_5, 'Pillar Scores for Top 5 Communities', 
                            'Step_6/output/figures/improved_radar_top5.png')
create_improved_radar_chart(bottom_5, 'Pillar Scores for Bottom 5 Communities', 
                            'Step_6/output/figures/improved_radar_bottom5.png')

# Create combined radar chart with top and bottom communities
plt.figure(figsize=(18, 9))

# Top 5 on left
plt.subplot(1, 2, 1, polar=True)
categories = ['Demographics', 'Income', 'Housing', 'Crime']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Configure axes
plt.xticks(angles[:-1], categories, size=14)
plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], size=10)
plt.ylim(0, 1)

# Plot each top community
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

colors = plt.colormaps['plasma']
color_values = np.linspace(0, 0.8, len(bottom_5))

# Plot each bottom community
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
plt.savefig('Step_6/output/figures/combined_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Creating rank shift visualizations...")

# Create rank shift visualization
# Get communities with largest rank shifts
top_rank_shifts = composite_scores.sort_values('RankDiff_Equal_PCA', ascending=False).head(10)

plt.figure(figsize=(14, 8))
plt.hlines(y=top_rank_shifts['communityname'], 
           xmin=top_rank_shifts['CRI_EqualWeights_Rank'], 
           xmax=top_rank_shifts['CRI_PCAWeights_Rank'], 
           color='blue', alpha=0.7, linewidth=2.5)

# Add markers for ranks
plt.scatter(top_rank_shifts['CRI_EqualWeights_Rank'], top_rank_shifts['communityname'], 
            color='navy', s=100, label='Equal Weights', zorder=10)
plt.scatter(top_rank_shifts['CRI_PCAWeights_Rank'], top_rank_shifts['communityname'], 
            color='orangered', s=100, label='PCA Weights', zorder=10)

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

# Create a second rank shift visualization for Equal vs. Stakeholder
top_stakeholder_shifts = composite_scores.sort_values('RankDiff_Equal_Stakeholder', ascending=False).head(10)

plt.figure(figsize=(14, 8))
plt.hlines(y=top_stakeholder_shifts['communityname'], 
           xmin=top_stakeholder_shifts['CRI_EqualWeights_Rank'], 
           xmax=top_stakeholder_shifts['CRI_StakeholderWeights_Rank'], 
           color='green', alpha=0.7, linewidth=2.5)

plt.scatter(top_stakeholder_shifts['CRI_EqualWeights_Rank'], top_stakeholder_shifts['communityname'], 
            color='navy', s=100, label='Equal Weights', zorder=10)
plt.scatter(top_stakeholder_shifts['CRI_StakeholderWeights_Rank'], top_stakeholder_shifts['communityname'], 
            color='crimson', s=100, label='Stakeholder Weights', zorder=10)

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

print("Improved visualizations created successfully!") 