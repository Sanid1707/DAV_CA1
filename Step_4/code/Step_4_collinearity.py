#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Multivariate Analysis - Section F & G (Collinearity Check & Indicator Selection)
- Check collinearity between indicators within each pillar
- Create correlation-based dendrograms
- Create initial decision grid for indicator selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import os
import glob

# Create directories if they don't exist
os.makedirs('Step_4/output/figures', exist_ok=True)
os.makedirs('Step_4/data', exist_ok=True)

print("\n" + "="*80)
print("SECTION 4-F & 4-G: INDICATOR COLLINEARITY CHECK & SELECTION")
print("="*80)

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

# Utility function to calculate communality
def calculate_communality(var, pillar, loadings_df, num_components):
    """Calculate communality for a variable based on PCA loadings."""
    if pillar not in loadings_df:
        return 0
    
    var_loadings = loadings_df[pillar][loadings_df[pillar]['Variable'] == var]
    if var_loadings.empty:
        return 0
    
    # Sum squared loadings for the kept components
    communality = 0
    for i in range(1, num_components + 1):
        loading_col = f'PC{i}'
        if loading_col in var_loadings.columns:
            communality += var_loadings[loading_col].values[0] ** 2
    
    return communality

# Try to load the final dataset from Step 3
# Look in multiple possible locations
possible_paths = [
    '../step3_final_dataset_for_multivariate.csv',
    '../../step3_final_dataset_for_multivariate.csv',
    '/Users/sanidhyapandey/DAV_CA1/step3_final_dataset_for_multivariate.csv',
    '../Step_3/data/step3_imputed_dataset.csv',
    '../../Step_3/data/step3_imputed_dataset.csv'
]

# Use glob to find any step3 final dataset file
glob_patterns = [
    '../*step3*final*dataset*.csv',
    '../../*step3*final*dataset*.csv',
    '../output/data/*step*dataset*.csv',
    '../Step_3/data/*step3*dataset*.csv'
]

for pattern in glob_patterns:
    possible_paths.extend(glob.glob(pattern))

# Try to load from any of the possible paths
df = None
for path in possible_paths:
    try:
        df = pd.read_csv(path)
        print(f"Loaded data from {path}")
        print(f"Dataset contains {df.shape[0]} communities and {df.shape[1]} variables\n")
        break
    except:
        continue

if df is None:
    print("Error: Could not find the dataset file. Please specify the correct path.")
    print("Tried the following paths:")
    for path in possible_paths:
        print(f"- {path}")
    exit(1)

# Store collinearity results
collinearity_results = {}
indicator_decisions = []

# 4-F: Cluster analysis on indicators (collinearity check)
print("\n4-F: Performing collinearity check on indicators...")

# Try to load PCA loadings from previous steps if available
pca_available = False
pca_loadings = {}
try:
    # Load PCA results
    for pillar in pillars.keys():
        try:
            pca_loadings[pillar] = pd.read_csv(f'Step_4/data/pca_loadings_{pillar}.csv')
        except:
            pass
    pca_available = True
    print("PCA loadings loaded from previous analysis")
except:
    print("Warning: PCA loadings not found, using correlation-based communality estimates")

# Analyze each pillar for collinearity
for pillar, variables in pillars.items():
    print(f"\nAnalyzing collinearity in {pillar} pillar...")
    
    # Check if we have enough variables
    valid_vars = [var for var in variables if var in df.columns]
    if len(valid_vars) < 2:
        print(f"  Not enough variables in {pillar} pillar for collinearity analysis. Skipping.")
        continue
    
    # Create correlation matrix
    corr_matrix = df[valid_vars].corr()
    
    # Save correlation matrix
    corr_matrix.to_csv(f'Step_4/output/{pillar}_correlation_matrix.csv')
    
    # Convert correlation to distance (1 - |correlation|)
    dist_matrix = 1 - np.abs(corr_matrix)
    
    # Perform hierarchical clustering on indicators
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(16, 10))
    plt.title(f'Indicator Correlation Dendrogram - {pillar}', fontsize=18)
    dendrogram(
        linkage_matrix,
        labels=valid_vars,
        orientation='right',
        leaf_font_size=12
    )
    plt.axvline(x=0.5, color='r', linestyle='--', label='Cutoff for high correlation')
    plt.xlabel('Distance (1 - |correlation|)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/{pillar}_indicator_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate heatmap with hierarchical clustering
    plt.figure(figsize=(16, 14))
    clustered_indices = dendrogram(linkage_matrix, no_plot=True)['leaves']
    clustered_corr = corr_matrix.iloc[clustered_indices, clustered_indices]
    
    mask = np.triu(np.ones_like(clustered_corr, dtype=bool))
    
    ax = sns.heatmap(
        clustered_corr, 
        cmap='coolwarm', 
        center=0, 
        annot=True, 
        fmt='.2f', 
        linewidths=0.5,
        mask=mask
    )
    
    # Identify highly correlated pairs (|r| > 0.95)
    high_corr_pairs = []
    for i in range(len(valid_vars)):
        for j in range(i + 1, len(valid_vars)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((
                    valid_vars[i], 
                    valid_vars[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    # Add highly correlated pairs to the plot
    if high_corr_pairs:
        plt.figtext(
            0.5, 0.01, 
            f"Highly correlated pairs (|r| > 0.95): {', '.join([f'{p[0]}-{p[1]} ({p[2]:.2f})' for p in high_corr_pairs])}", 
            ha='center', 
            fontsize=12, 
            bbox=dict(facecolor='yellow', alpha=0.2)
        )
    
    plt.title(f'Correlation Heatmap with Clustering - {pillar}', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f'Step_4/output/figures/{pillar}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Store collinearity results
    collinearity_results[pillar] = {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'linkage_matrix': linkage_matrix
    }
    
    print(f"  Collinearity analysis for {pillar} completed")
    print(f"  Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.95)")
    
    # 4-G: Decision grid for indicators
    print(f"\n4-G: Creating decision grid for {pillar} indicators...")
    
    # Determine number of components to keep from PCA
    if pca_available:
        # Get number of components for this pillar
        try:
            eigenvalue_df = pd.read_csv('Step_4/data/pca_eigenvalues_summary.csv')
            components_to_keep = eigenvalue_df[(eigenvalue_df['Pillar'] == pillar) & 
                                            (eigenvalue_df['Keep'] == True)].shape[0]
        except:
            # Default if can't load eigenvalues
            components_to_keep = 2
    else:
        # Default to 2 components if PCA results not available
        components_to_keep = 2
    
    print(f"  Using {components_to_keep} principal components for communality calculation")
    
    # Create decision table for each indicator
    for var in valid_vars:
        # Calculate communality if PCA is available
        if pca_available:
            communality = calculate_communality(var, pillar, pca_loadings, components_to_keep)
        else:
            # Approximate communality as maximum squared correlation with any other variable
            max_corr = 0
            for other_var in valid_vars:
                if other_var != var:
                    max_corr = max(max_corr, corr_matrix.loc[var, other_var]**2)
            communality = max_corr
        
        # Check if this variable is in a high correlation pair
        in_high_corr_pair = any(var in (p[0], p[1]) for p in high_corr_pairs)
        
        # Get correlation partners if in a high correlation pair
        corr_partners = []
        if in_high_corr_pair:
            for p in high_corr_pairs:
                if var == p[0]:
                    corr_partners.append((p[1], p[2]))
                elif var == p[1]:
                    corr_partners.append((p[0], p[2]))
        
        # Make initial decision
        if communality < 0.5:
            decision = "Drop (low communality)"
            rationale = f"Low communality ({communality:.2f}), contributes little to principal components."
        elif in_high_corr_pair:
            # Check if this variable has better communality than its partners
            better_communality = True
            for partner, _ in corr_partners:
                if pca_available:
                    partner_communality = calculate_communality(partner, pillar, pca_loadings, components_to_keep)
                else:
                    # Approximate
                    max_corr = 0
                    for other_var in valid_vars:
                        if other_var != partner:
                            max_corr = max(max_corr, corr_matrix.loc[partner, other_var]**2)
                    partner_communality = max_corr
                
                if partner_communality > communality:
                    better_communality = False
                    break
            
            if better_communality:
                decision = "Keep (representative of group)"
                rationale = f"High correlation with {', '.join([p[0] for p in corr_partners])}, but highest communality in group."
            else:
                decision = "Consider merging or drop"
                rationale = f"High correlation with {', '.join([p[0] for p in corr_partners])}, consider merging or using partner with higher communality."
        else:
            decision = "Keep (unique information)"
            rationale = f"Good communality ({communality:.2f}) and relatively independent from other indicators."
        
        # Add to decision table
        indicator_decisions.append({
            'Pillar': pillar,
            'Indicator': var,
            'Communality': communality,
            'In_High_Correlation_Group': in_high_corr_pair,
            'Correlation_Partners': '; '.join([f"{p[0]} ({p[1]:.2f})" for p in corr_partners]),
            'Decision': decision,
            'Rationale': rationale
        })

# Create decision grid dataframe
decision_df = pd.DataFrame(indicator_decisions)
decision_df.to_csv('Step_4/output/indicator_decision_grid.csv', index=False)
print("\nIndicator decision grid saved to 'Step_4/output/indicator_decision_grid.csv'")

# Generate markdown report for indicator decisions
decision_report = "# Indicator Selection Decision Grid\n\n"
decision_report += "## Summary\n\n"

# Add summary statistics
decision_report += f"- Total indicators analyzed: {len(indicator_decisions)}\n"
decision_report += f"- Indicators to keep: {sum(1 for d in indicator_decisions if d['Decision'].startswith('Keep'))}\n"
decision_report += f"- Indicators to consider merging: {sum(1 for d in indicator_decisions if 'merging' in d['Decision'])}\n"
decision_report += f"- Indicators to drop: {sum(1 for d in indicator_decisions if d['Decision'].startswith('Drop'))}\n\n"

# Add tables for each pillar
for pillar in pillars.keys():
    pillar_decisions = [d for d in indicator_decisions if d['Pillar'] == pillar]
    
    if not pillar_decisions:
        continue
    
    decision_report += f"## {pillar} Pillar\n\n"
    decision_report += "| Indicator | Communality | High Correlation | Decision | Rationale |\n"
    decision_report += "|-----------|-------------|------------------|----------|----------|\n"
    
    for decision in pillar_decisions:
        decision_report += f"| {decision['Indicator']} | {decision['Communality']:.2f} | "
        decision_report += f"{'Yes' if decision['In_High_Correlation_Group'] else 'No'} | "
        decision_report += f"{decision['Decision']} | {decision['Rationale']} |\n"
    
    decision_report += "\n"

# Save decision report
with open('Step_4/output/indicator_decision_report.md', 'w') as f:
    f.write(decision_report)

print("Indicator decision report saved to 'Step_4/output/indicator_decision_report.md'")
print("\nCollinearity analysis and indicator selection completed successfully!") 