#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Correlation Plots for Step 6

This script generates enhanced correlation plots for the four pillars.
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

# Load normalized data from Step 5
normalized_data = pd.read_csv('Step_5/output/step5_normalized_dataset.csv')

# Define the pillars and their respective indicators
pillars = {
    'Demographics': ['racepctblack', 'racePctHisp', 'pctUrban', 'PctNotSpeakEnglWell'],
    'Income': ['medIncome', 'pctWPubAsst', 'PctPopUnderPov', 'PctUnemployed'],
    'Housing': ['PctFam2Par', 'PctHousOccup', 'PctVacantBoarded', 'PctHousNoPhone', 'PctSameHouse85'],
    'Crime': ['murdPerPop', 'robbbPerPop', 'autoTheftPerPop', 'arsonsPerPop', 'ViolentCrimesPerPop']
}

# Create improved correlation plots for each pillar
for pillar_name, indicators in pillars.items():
    print(f"Creating improved correlation plot for {pillar_name}...")
    
    # Calculate correlation matrix
    corr_matrix = normalized_data[indicators].corr()
    
    # Check for highly correlated pairs (|r| > 0.9)
    high_corr_pairs = []
    for i in range(len(indicators)):
        for j in range(i+1, len(indicators)):
            var1 = indicators[i]
            var2 = indicators[j]
            corr_value = corr_matrix.loc[var1, var2]
            if abs(corr_value) > 0.9:
                high_corr_pairs.append((var1, var2, corr_value))
    
    # Create heatmap with better formatting
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                mask=mask, fmt='.2f', linewidths=0.5, square=True)
    
    # Add title and subtitle
    plt.title(f'Correlation Matrix: {pillar_name} Pillar', fontsize=16, pad=20)
    
    # Add annotation for high correlations if any
    if high_corr_pairs:
        plt.figtext(0.5, 0.01, 
                    f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9)",
                    ha="center", fontsize=12, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(f'Step_6/output/figures/improved_corr_{pillar_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create a single figure with all four correlation matrices
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

print("Improved correlation plots created successfully!") 