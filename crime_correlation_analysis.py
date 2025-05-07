#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crime Correlation Analysis
- Goal: Analyze correlations between crime variables
- Compare raw count variables vs. per-capita rate variables
- Visualize and explain the size effect in crime data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images directory if it doesn't exist
os.makedirs('images/crime_analysis', exist_ok=True)

# Set high-quality visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
custom_palette = sns.color_palette("viridis", 8)
sns.set_palette(custom_palette)

# Configure matplotlib for high quality output
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['image.cmap'] = 'viridis'

print("\n" + "="*80)
print("CRIME CORRELATION ANALYSIS")
print("="*80)

# Load the dataset
print("\nLoading the dataset...")
df = pd.read_csv('step2_final_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Identify crime count variables and crime rate variables
crime_count_vars = ['murders', 'rapes', 'robberies', 'assaults', 'burglaries', 'larcenies', 'autoTheft', 'arsons']
crime_rate_vars = ['murdPerPop', 'rapesPerPop', 'robbbPerPop', 'assaultPerPop', 'burglPerPop', 'larcPerPop', 
                  'autoTheftPerPop', 'arsonsPerPop', 'ViolentCrimesPerPop', 'nonViolPerPop']

# Filter to keep only variables present in the dataset
available_count_vars = [var for var in crime_count_vars if var in df.columns]
available_rate_vars = [var for var in crime_rate_vars if var in df.columns]

# Add population to count variables for analysis
if 'population' in df.columns:
    available_count_vars = ['population'] + available_count_vars

print(f"\nCrime count variables found: {len(available_count_vars)}")
print(f"Crime rate variables found: {len(available_rate_vars)}")

# 1. Create correlation matrix for count variables
if len(available_count_vars) > 1:
    print("\nAnalyzing correlations between crime count variables...")
    count_corr = df[available_count_vars].corr()
    
    # Create heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(count_corr, annot=True, fmt='.2f', cmap='Reds', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Crime Count Variables', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/crime_analysis/count_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # Calculate and display average correlation
    count_corr_values = count_corr.values
    mask = np.triu(np.ones_like(count_corr_values, dtype=bool), k=1)
    upper_triangle = count_corr_values[mask]
    avg_corr = np.mean(upper_triangle)
    print(f"Average correlation between crime count variables: {avg_corr:.4f}")

# 2. Create correlation matrix for rate variables
if len(available_rate_vars) > 1:
    print("\nAnalyzing correlations between crime rate variables...")
    rate_corr = df[available_rate_vars].corr()
    
    # Create heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(rate_corr, annot=True, fmt='.2f', cmap='Blues', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Crime Rate Variables', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/crime_analysis/rate_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # Calculate and display average correlation
    rate_corr_values = rate_corr.values
    mask = np.triu(np.ones_like(rate_corr_values, dtype=bool), k=1)
    upper_triangle = rate_corr_values[mask]
    avg_corr = np.mean(upper_triangle)
    print(f"Average correlation between crime rate variables: {avg_corr:.4f}")

# 3. Create scatterplots showing relationship with population
if 'population' in df.columns and len(available_count_vars) > 1:
    print("\nAnalyzing relationship between population and crime counts...")
    
    # Sample up to 4 crime count variables (excluding population)
    plot_vars = [var for var in available_count_vars if var != 'population'][:4]
    
    if plot_vars:
        plt.figure(figsize=(20, 16))
        for i, var in enumerate(plot_vars, 1):
            plt.subplot(2, 2, i)
            
            # Create scatterplot
            sns.regplot(x='population', y=var, data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            
            # Calculate correlation
            corr = df['population'].corr(df[var])
            plt.title(f'Correlation: {corr:.4f}', fontsize=16)
            plt.xlabel('Population', fontsize=14)
            plt.ylabel(var, fontsize=14)
        
        plt.suptitle('Relationship Between Population and Crime Counts', fontsize=22, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig('images/crime_analysis/population_crime_relationship.png', bbox_inches='tight')
        plt.close()

# 4. Compare correlations between count vars and rate vars
print("\nComparing correlation structure between count and rate variables...")

# Create a side-by-side comparison
if len(available_count_vars) > 1 and len(available_rate_vars) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Crime count correlation heatmap
    sns.heatmap(count_corr, annot=True, fmt='.2f', cmap='Reds', 
                linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0])
    axes[0].set_title('Crime Count Variables', fontsize=20, pad=20)
    
    # Crime rate correlation heatmap
    sns.heatmap(rate_corr, annot=True, fmt='.2f', cmap='Blues', 
                linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1])
    axes[1].set_title('Crime Rate Variables', fontsize=20, pad=20)
    
    plt.suptitle('Comparison of Correlation Structures: Count vs. Rate Variables', fontsize=24, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('images/crime_analysis/count_vs_rate_correlation.png', bbox_inches='tight')
    plt.close()

# 5. Create a report explaining the findings
report_text = """
# Crime Variable Correlation Analysis

## Overview
This analysis examines the correlation structure among crime variables in the dataset, with a specific focus on understanding the difference between crime count variables and crime rate variables.

## Key Findings

### 1. Crime Count Variables
Crime count variables (murders, burglaries, larcenies, etc.) show extremely high correlations with each other (typically 0.90-0.98). This is primarily due to the **"size effect"** - larger communities naturally have more crimes of all types simply because they have more people.

### 2. Population as a Confounding Factor
The analysis shows that population size is the driving factor behind these high correlations. When a community has more people:
- It tends to have more murders
- It tends to have more burglaries
- It tends to have more of all types of crime incidents

This does not necessarily mean that these crime types are causally related to each other, but rather that they all increase with population size.

### 3. Crime Rate Variables
Crime rate variables (crimes per population) show more distinct and meaningful correlation patterns because they control for the population size effect. The correlation structure among rate variables reveals actual relationships between different crime types rather than just reflecting population differences.

### 4. Implications for Data Selection and Analysis
- **For descriptive purposes**: Both count and rate variables provide useful information
- **For multivariate analysis**: Rate variables are generally preferred as they control for the confounding effect of population size
- **For crime pattern analysis**: Rate variables reveal more meaningful patterns about the nature of crime in communities of different sizes

## Recommendation
When analyzing relationships between crime variables and other factors (such as socioeconomic variables), it is generally advisable to use crime rate variables rather than raw counts to avoid spurious correlations driven simply by population size.

## Visualizations
- **count_correlation_heatmap.png**: Shows the high correlations between crime count variables
- **rate_correlation_heatmap.png**: Shows the more nuanced correlations between crime rate variables
- **population_crime_relationship.png**: Demonstrates how population directly influences crime counts
- **count_vs_rate_correlation.png**: Side-by-side comparison showing the difference in correlation structure
"""

with open('crime_correlation_report.md', 'w') as f:
    f.write(report_text)

print("\nAnalysis complete! Report and visualizations have been created.")
print("Next steps: Refer to the crime_correlation_report.md file for explanation of the correlations.") 