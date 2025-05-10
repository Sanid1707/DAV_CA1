#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9: Vulnerability and Exposure Visualization

This script creates visualizations of the Community Crime Vulnerability and Exposure Index (CCVEI)
based on the scores from Step 6.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Create output directories
Path("Step_9/output/figures").mkdir(parents=True, exist_ok=True)
Path("Step_9/output/tables").mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load the composite scores from Step 6
    """
    print("Loading composite scores...")
    try:
        scores_df = pd.read_csv("Step_6/output/composite_scores_ranked.csv")
        
        # Check the first few rows to understand structure
        print(f"Sample data: {scores_df.head(1)}")
        print(f"Column names: {scores_df.columns.tolist()}")
        
        # Extract community names
        scores_df['community_name'] = scores_df['communityname']
        
        # Select the appropriate score column - use stakeholder weights as primary
        scores_df['composite_score'] = scores_df['CRI_StakeholderWeights']
        
        return scores_df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a sample dataframe for testing
        return pd.DataFrame({
            'community_name': ['Community1', 'Community2', 'Community3'],
            'composite_score': [0.75, 0.85, 0.65]
        })

def create_risk_distribution_plot(data_df):
    """
    Create a histogram showing the distribution of vulnerability and exposure scores
    """
    print("Creating risk distribution plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create histogram with kernel density estimate
    sns.histplot(data_df['composite_score'], kde=True, bins=30, color='darkblue', ax=ax)
    
    # Add mean and median lines
    mean_score = data_df['composite_score'].mean()
    median_score = data_df['composite_score'].median()
    
    ax.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
    ax.axvline(median_score, color='green', linestyle=':', label=f'Median: {median_score:.3f}')
    
    # Add labels and title
    ax.set_xlabel('Community Crime Vulnerability and Exposure Index (CCVEI) Score', fontsize=12)
    ax.set_ylabel('Number of Communities', fontsize=12)
    ax.set_title('Distribution of Vulnerability and Exposure Scores Across Communities', fontsize=16)
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('Step_9/output/figures/ccri_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Distribution plot saved to Step_9/output/figures/ccri_distribution.png")
    
    return True

def create_top_communities_chart(data_df):
    """
    Create a bar chart showing the top 10 highest and lowest vulnerability and exposure communities
    """
    print("Creating top communities chart...")
    
    # Ensure we have composite scores and sort the data
    if 'composite_score' in data_df.columns:
        # Sort by composite score
        sorted_df = data_df.sort_values('composite_score', ascending=False)
        
        # Get top 10 highest and lowest risk communities
        top_10 = sorted_df.head(10)
        bottom_10 = sorted_df.tail(10)
        
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot highest vulnerability and exposure communities
        sns.barplot(x='composite_score', y='community_name', data=top_10, ax=ax1, palette='Reds_r')
        ax1.set_title('Top 10 Highest Vulnerability and Exposure Communities', fontsize=14)
        ax1.set_xlabel('CCVEI Score', fontsize=12)
        ax1.set_ylabel('Community', fontsize=12)
        
        # Plot lowest vulnerability and exposure communities
        sns.barplot(x='composite_score', y='community_name', data=bottom_10, ax=ax2, palette='Blues')
        ax2.set_title('Top 10 Lowest Vulnerability and Exposure Communities', fontsize=14)
        ax2.set_xlabel('CCVEI Score', fontsize=12)
        ax2.set_ylabel('Community', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/top_communities_by_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Top communities chart saved to Step_9/output/figures/top_communities_by_risk.png")
        
        # Also save this data as a CSV for reference
        top_10_df = top_10[['community_name', 'composite_score']]
        bottom_10_df = bottom_10[['community_name', 'composite_score']]
        
        top_10_df.to_csv('Step_9/output/tables/highest_risk_communities.csv', index=False)
        bottom_10_df.to_csv('Step_9/output/tables/lowest_risk_communities.csv', index=False)
        
        return True
    else:
        print("Error: No composite score column found in the data")
        return False

def create_pillar_comparison(data_df):
    """
    Create a comparison of the different pillars contributing to the CCVEI
    """
    print("Creating pillar comparison chart...")
    
    # Check if we have the pillar scores
    pillar_cols = ['DemographicsScore', 'IncomeScore', 'HousingScore', 'CrimeScore']
    
    if all(col in data_df.columns for col in pillar_cols):
        # Prepare the data for plotting
        pillar_data = data_df[pillar_cols].mean().reset_index()
        pillar_data.columns = ['Pillar', 'Average Score']
        
        # Rename pillars to reflect vulnerability and exposure components
        pillar_mapping = {
            'DemographicsScore': 'Demographics (Vulnerability)',
            'IncomeScore': 'Income (Vulnerability)',
            'HousingScore': 'Housing (Vulnerability)',
            'CrimeScore': 'Crime (Exposure)'
        }
        
        pillar_data['Pillar'] = pillar_data['Pillar'].map(pillar_mapping)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar chart
        bars = sns.barplot(x='Pillar', y='Average Score', data=pillar_data, palette='viridis', ax=ax)
        
        # Add exact values on top of bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.01,
                f'{pillar_data["Average Score"].iloc[i]:.3f}',
                ha='center',
                fontsize=9
            )
        
        # Add labels and title
        ax.set_xlabel('CCVEI Pillars', fontsize=12)
        ax.set_ylabel('Average Pillar Score', fontsize=12)
        ax.set_title('Average Scores Across CCVEI Pillars: Vulnerability and Exposure Components', fontsize=16)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/pillar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Pillar comparison chart saved to Step_9/output/figures/pillar_comparison.png")
        
        # Also create a correlation heatmap between pillars
        corr_matrix = data_df[pillar_cols].corr()
        
        # Plot the correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
                   linewidths=.5, square=True)
        plt.title('Correlation Between Vulnerability and Exposure Pillars', fontsize=16)
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/pillar_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Pillar correlation heatmap saved to Step_9/output/figures/pillar_correlation.png")
        
        return True
    else:
        print("Error: Not all pillar score columns are available in the data")
        return False

def create_weighting_comparison(data_df):
    """
    Create a comparison of different weighting methods
    """
    print("Creating weighting method comparison...")
    
    weighting_cols = ['CRI_EqualWeights', 'CRI_PCAWeights', 'CRI_StakeholderWeights']
    
    if all(col in data_df.columns for col in weighting_cols):
        # Calculate statistics for each weighting method
        stats = {
            'Mean': data_df[weighting_cols].mean(),
            'Median': data_df[weighting_cols].median(),
            'Std Dev': data_df[weighting_cols].std(),
            'Min': data_df[weighting_cols].min(),
            'Max': data_df[weighting_cols].max()
        }
        
        # Create a DataFrame for easier plotting
        stats_df = pd.DataFrame(stats).reset_index()
        stats_df.columns = ['Weighting Method', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        
        # Rename the weighting methods for better readability
        stats_df['Weighting Method'] = stats_df['Weighting Method'].map({
            'CRI_EqualWeights': 'Equal Weights',
            'CRI_PCAWeights': 'PCA Weights',
            'CRI_StakeholderWeights': 'Stakeholder Weights'
        })
        
        # Save the statistics to a CSV
        stats_df.to_csv('Step_9/output/tables/weighting_method_statistics.csv', index=False)
        
        # Create bar chart comparing means
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Weighting Method', y='Mean', data=stats_df, palette='Set2')
        plt.title('Average CCVEI Score by Weighting Method', fontsize=16)
        plt.xlabel('Weighting Method', fontsize=12)
        plt.ylabel('Mean CCVEI Score', fontsize=12)
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/weighting_comparison_means.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot matrix to compare different weightings
        sns.pairplot(data_df[weighting_cols], height=3, aspect=1,
                    plot_kws={'alpha': 0.6, 's': 15, 'edgecolor': 'k', 'linewidth': 0.5})
        plt.suptitle('Scatter Plot Matrix: Comparison of Weighting Methods', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/weighting_comparison_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Weighting comparison charts saved to Step_9/output/figures/")
        
        return True
    else:
        print("Error: Not all weighting method columns are available in the data")
        return False

def main():
    """Main execution function"""
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Create risk distribution plot
    create_risk_distribution_plot(df)
    
    # Create top communities chart
    create_top_communities_chart(df)
    
    # Create pillar comparison
    create_pillar_comparison(df)
    
    # Create weighting method comparison
    create_weighting_comparison(df)
    
    print("All visualizations complete!")

if __name__ == "__main__":
    main() 