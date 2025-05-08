#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Multivariate Analysis - Section E (Cluster Profiling)
- Profile clusters based on original indicators
- Create visualizations to interpret cluster differences
- Generate summary statistics for clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

# Create directories if they don't exist
os.makedirs('Step_4/output/figures', exist_ok=True)

print("\n" + "="*80)
print("SECTION 4-E: CLUSTER PROFILING")
print("="*80)

# Load data
try:
    # Load the component scores with cluster assignments
    all_components_df = pd.read_csv('Step_4/data/all_component_scores_with_clusters.csv')
    print(f"Loaded component scores with {all_components_df.shape[0]} communities")
    
    # Load the original data
    df = pd.read_csv('step3_final_dataset_for_multivariate.csv')
    print(f"Loaded original data with {df.shape[0]} communities and {df.shape[1]} variables")
except FileNotFoundError:
    try:
        # Alternative paths
        all_components_df = pd.read_csv('all_component_scores_with_clusters.csv')
        df = pd.read_csv('step3_imputed_dataset.csv')
        print(f"Loaded data from alternative paths")
    except:
        print("Error: Could not load required data files")
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

# Dictionary to store clustering results
cluster_profiles = {}

# 4-E: Profile clusters (using the original indicators)
print("\n4-E: Profiling clusters using original indicators...")

# Combine original data with cluster assignments
# First, ensure we have communityname in both dataframes for merging
merged_df = df.copy()

# Check if communityname columns exist for merging
if 'communityname' in merged_df.columns and 'communityname' in all_components_df.columns:
    # Merge on communityname
    merged_df = pd.merge(
        merged_df, 
        all_components_df[[col for col in all_components_df.columns if 'Cluster' in col or col == 'communityname']], 
        on='communityname'
    )
    print("Merged datasets using communityname")
else:
    # Assume same order if communityname not available
    cluster_cols = [col for col in all_components_df.columns if 'Cluster' in col]
    for col in cluster_cols:
        merged_df[col] = all_components_df[col].values
    print("Added cluster assignments assuming same data order (communityname not available)")

# Profile clusters for each pillar
for pillar, variables in pillars.items():
    # Check if we have clusters for this pillar
    kmeans_cluster_col = f'{pillar}_Cluster_KMeans'
    hierarch_cluster_col = f'{pillar}_Cluster_Hierarchical'
    
    if kmeans_cluster_col not in merged_df.columns:
        print(f"No cluster assignments found for {pillar}. Skipping profiling.")
        continue
    
    print(f"\nProfiling clusters for {pillar} pillar...")
    
    # Use k-means clusters for profiling (typically more stable than hierarchical)
    cluster_col = kmeans_cluster_col
    n_clusters = merged_df[cluster_col].nunique()
    
    print(f"  Found {n_clusters} clusters")
    
    # Filter valid variables (those that exist in the dataset)
    valid_vars = [var for var in variables if var in merged_df.columns]
    
    if not valid_vars:
        print(f"  No valid variables found for {pillar}. Skipping profiling.")
        continue
    
    # 1. Calculate mean values for each original indicator by cluster
    cluster_means = merged_df.groupby(cluster_col)[valid_vars].mean()
    
    # Store the cluster profiles
    cluster_profiles[pillar] = cluster_means
    
    # Save cluster profiles to CSV
    cluster_means.to_csv(f'Step_4/output/{pillar}_cluster_profiles.csv')
    print(f"  Cluster profiles saved to 'Step_4/output/{pillar}_cluster_profiles.csv'")
    
    # 2. Create visualizations of cluster profiles
    
    # Box plots for each variable by cluster
    n_vars = len(valid_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(18, 5 * n_rows))
    
    for i, var in enumerate(valid_vars):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x=cluster_col, y=var, data=merged_df)
        plt.title(f'{var} by Cluster', fontsize=14)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel(var, fontsize=12)
        plt.xticks(rotation=0)
        
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/{pillar}_cluster_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create radar plots for cluster profiles
    
    # Normalize the means for radar plot
    normalized_means = cluster_means.copy()
    for var in valid_vars:
        min_val = merged_df[var].min()
        max_val = merged_df[var].max()
        if max_val > min_val:
            normalized_means[var] = (cluster_means[var] - min_val) / (max_val - min_val)
    
    # Radar plot
    plt.figure(figsize=(15, 12))
    
    # Set up the radar plot
    angles = np.linspace(0, 2*np.pi, len(valid_vars), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Complete the variables list for the plot
    radar_vars = valid_vars + [valid_vars[0]]  # Close the circle
    
    # Set up the plot
    ax = plt.subplot(111, polar=True)
    
    # Add labels
    plt.xticks(angles[:-1], valid_vars, fontsize=12)
    
    # Draw each cluster
    for cluster in range(n_clusters):
        values = normalized_means.iloc[cluster].values.tolist()
        values += values[:1]  # Close the circle
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    plt.title(f'Radar Plot of Cluster Profiles - {pillar}', fontsize=18, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/{pillar}_cluster_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create heatmap of cluster profiles
    plt.figure(figsize=(16, 10))
    
    # Normalize data for heatmap
    normalized_for_heatmap = normalized_means.copy()
    
    # Generate heatmap
    sns.heatmap(normalized_for_heatmap, cmap='viridis', annot=True, fmt='.2f', linewidths=0.5)
    plt.title(f'Cluster Profiles Heatmap - {pillar}', fontsize=18)
    plt.xlabel('Variables', fontsize=16)
    plt.ylabel('Cluster', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/{pillar}_cluster_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generate cluster description
    cluster_description = f"# Cluster Profiles for {pillar} Pillar\n\n"
    cluster_description += f"## Overview\n"
    cluster_description += f"- Number of clusters: {n_clusters}\n"
    cluster_description += f"- Variables analyzed: {len(valid_vars)}\n\n"
    
    for cluster in range(n_clusters):
        cluster_description += f"## Cluster {cluster}\n\n"
        cluster_description += f"### Key Characteristics:\n\n"
        
        # Get top 3 highest and lowest values relative to overall mean
        relative_means = (cluster_means.iloc[cluster] - merged_df[valid_vars].mean()) / merged_df[valid_vars].std()
        top_high = relative_means.nlargest(3)
        top_low = relative_means.nsmallest(3)
        
        cluster_description += f"**Distinctive high values:**\n"
        for var, val in top_high.items():
            cluster_description += f"- {var}: {cluster_means.iloc[cluster][var]:.2f} ({val:+.2f} standard deviations from mean)\n"
        
        cluster_description += f"\n**Distinctive low values:**\n"
        for var, val in top_low.items():
            cluster_description += f"- {var}: {cluster_means.iloc[cluster][var]:.2f} ({val:+.2f} standard deviations from mean)\n"
        
        cluster_description += "\n"
    
    # Save cluster description to file
    with open(f'Step_4/output/{pillar}_cluster_description.md', 'w') as f:
        f.write(cluster_description)
    
    print(f"  Cluster description saved to 'Step_4/output/{pillar}_cluster_description.md'")

# Create a combined summary of all pillar clusters
all_clusters_summary = "# Combined Cluster Analysis Summary\n\n"

for pillar, profile in cluster_profiles.items():
    all_clusters_summary += f"## {pillar} Pillar\n\n"
    all_clusters_summary += f"Number of clusters: {profile.shape[0]}\n\n"
    
    all_clusters_summary += "### Brief Cluster Descriptions\n\n"
    
    for cluster in range(profile.shape[0]):
        # Get variables for this pillar
        valid_vars = [var for var in pillars[pillar] if var in merged_df.columns]
        
        # Get means for this cluster
        relative_means = (profile.iloc[cluster] - merged_df[valid_vars].mean()) / merged_df[valid_vars].std()
        
        # Get top 2 distinctive features
        distinctive_features = relative_means.abs().nlargest(2)
        
        all_clusters_summary += f"**Cluster {cluster}**: "
        feature_descriptions = []
        
        for var in distinctive_features.index:
            direction = "high" if relative_means[var] > 0 else "low"
            feature_descriptions.append(f"{direction} {var}")
        
        all_clusters_summary += f"Characterized by {' and '.join(feature_descriptions)}\n\n"
    
    all_clusters_summary += "\n"

# Save combined summary
with open('Step_4/output/all_clusters_summary.md', 'w') as f:
    f.write(all_clusters_summary)

print("\nCombined cluster summary saved to 'Step_4/output/all_clusters_summary.md'")
print("\nCluster profiling completed successfully!") 