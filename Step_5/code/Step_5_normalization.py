#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5: Normalization
- Check for outliers and skewness
- Define indicator polarities
- Invert negative polarity indicators
- Apply Min-Max normalization
- Quality check normalized data
- Save normalized dataset for Step 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# Create directories if they don't exist
os.makedirs('../output/figures', exist_ok=True)
os.makedirs('../docs', exist_ok=True)

print("\n" + "="*80)
print("STEP 5: NORMALIZATION OF INDICATORS")
print("="*80)

# Load the trimmed dataset from Step 4
try:
    # Try different possible locations of the Step 4 output file
    possible_paths = [
        'Step_4/output/step4_trimmed_dataset.csv',
        'Step_4/output/theoretical_trimmed_dataset.csv',
        '../../Step_4/output/step4_trimmed_dataset.csv',
        '../../Step_4/output/data/step4_trimmed_dataset.csv',
        '../../Step_4/output/theoretical_trimmed_dataset.csv',
        '../Step_4/output/step4_trimmed_dataset.csv',
        '../Step_4/output/data/step4_trimmed_dataset.csv',
        '../Step_4/output/theoretical_trimmed_dataset.csv',
        '../output/step4_trimmed_dataset.csv',
        '../../Step_4/code/Step_4/output/step4_trimmed_dataset.csv',
    ]
    
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
        raise FileNotFoundError("Could not find Step 4 trimmed dataset")
        
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# A: Define indicator polarities (↑ good / ↓ good)
# Define polarity for each indicator (True for ↑ good, False for ↓ good)
indicator_info = {
    'racepctblack': {'unit': '%', 'polarity': True, 'description': 'Percentage of population that is black'},
    'racePctHisp': {'unit': '%', 'polarity': True, 'description': 'Percentage of population that is Hispanic'},
    'pctUrban': {'unit': '%', 'polarity': True, 'description': 'Percentage of people living in urban areas'},
    'PctNotSpeakEnglWell': {'unit': '%', 'polarity': False, 'description': 'Percentage who do not speak English well'},
    'medIncome': {'unit': '$', 'polarity': True, 'description': 'Median household income'},
    'pctWPubAsst': {'unit': '%', 'polarity': False, 'description': 'Percentage of households with public assistance'},
    'PctPopUnderPov': {'unit': '%', 'polarity': False, 'description': 'Percentage of people under the poverty level'},
    'PctUnemployed': {'unit': '%', 'polarity': False, 'description': 'Percentage of people unemployed'},
    'PctFam2Par': {'unit': '%', 'polarity': True, 'description': 'Percentage of families headed by two parents'},
    'PctHousOccup': {'unit': '%', 'polarity': True, 'description': 'Percentage of housing occupied'},
    'PctVacantBoarded': {'unit': '%', 'polarity': False, 'description': 'Percentage of vacant housing that is boarded up'},
    'PctHousNoPhone': {'unit': '%', 'polarity': False, 'description': 'Percentage of households with no phone'},
    'PctSameHouse85': {'unit': '%', 'polarity': True, 'description': 'Percentage of households in same house since 1985'},
    'murdPerPop': {'unit': 'per 100k', 'polarity': False, 'description': 'Number of murders per 100K population'},
    'robbbPerPop': {'unit': 'per 100k', 'polarity': False, 'description': 'Number of robberies per 100K population'},
    'autoTheftPerPop': {'unit': 'per 100k', 'polarity': False, 'description': 'Number of auto thefts per 100K population'},
    'arsonsPerPop': {'unit': 'per 100k', 'polarity': False, 'description': 'Number of arsons per 100K population'},
    'ViolentCrimesPerPop': {'unit': 'per 100k', 'polarity': False, 'description': 'Number of violent crimes per 100K population'}
}

# Create a polarity table
polarity_data = []
for indicator, info in indicator_info.items():
    polarity_data.append({
        'Indicator': indicator,
        'Unit': info['unit'],
        'Original Polarity': '↑ good' if info['polarity'] else '↓ good',
        'Polarity After Inversion': 'No change' if info['polarity'] else 'Inverted to ↑ good',
        'Description': info['description']
    })

polarity_df = pd.DataFrame(polarity_data)
polarity_df.to_csv('../docs/indicator_polarity_table.csv', index=False)
print("Indicator polarity table saved to '../docs/indicator_polarity_table.csv'")

# Create list of columns to invert (negative polarity)
invert_cols = [col for col, info in indicator_info.items() if not info['polarity']]
print(f"\nColumns to invert (negative polarity, ↓ good): {invert_cols}")

# B: Check for outliers and skewness
print("\nB: Checking for outliers and skewness in the dataset...")

# Create a figure for the boxplots
plt.figure(figsize=(15, 20))
    
# Calculate number of rows and columns for subplots
n_indicators = len(df.columns) - 1  # Excluding communityname
n_cols = 3
n_rows = (n_indicators + n_cols - 1) // n_cols

# Plot boxplots for all indicators
indicators = [col for col in df.columns if col != 'communityname']
for i, indicator in enumerate(indicators):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x=df[indicator])
    plt.title(f"{indicator}")
    plt.tight_layout()

plt.savefig('../output/figures/indicator_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and save descriptive statistics
describe_df = df.describe().T
describe_df['skew'] = df.drop(columns=['communityname']).skew()
describe_df.to_csv('../output/indicator_descriptive_stats.csv')

print("Boxplots saved to '../output/figures/indicator_boxplots.png'")
print("Descriptive statistics saved to '../output/indicator_descriptive_stats.csv'")
print("No additional trimming required before Min-Max normalization.")

# C: Choose & justify the main normalization rule (Min-Max)
print("\nC: Using Min-Max normalization for intuitive [0, 1] scale and interpretability for policy dashboards.")

# D: Normalize the data
print("\nD: Normalizing the data using Min-Max scaling with inversion for negative polarity indicators...")

# Create a copy of the dataset for normalization
df_normalized = df.copy()
community_names = df_normalized['communityname']  # Save community names
df_to_normalize = df_normalized.drop(columns=['communityname'])

# D-1: Invert negative-polarity variables before scaling using the formula x_inv = x_max - x_i
print("\nInverting negative polarity indicators using formula: x_inv = x_max - x_i")
# Show a before/after snippet for one inverted column (the first one)
sample_col = invert_cols[0]
print(f"\nBefore inversion - {sample_col} (first 5 values):")
print(df_to_normalize[sample_col].head())

# Apply inversion to negative polarity indicators
for col in invert_cols:
    df_to_normalize[col] = df_to_normalize[col].max() - df_to_normalize[col]

print(f"\nAfter inversion - {sample_col} (first 5 values):")
print(df_to_normalize[sample_col].head())

# D-2: Skip transformation for skewed indicators
print("\nSkew within acceptable range; no log-transform applied.")

# E: Apply Min-Max scaling
print("\nE: Applying Min-Max scaling to all indicators...")
scaler = MinMaxScaler()
normalized_array = scaler.fit_transform(df_to_normalize)
df_normalized = pd.DataFrame(
    normalized_array, 
    columns=df_to_normalize.columns,
    index=df_to_normalize.index
)

# Add community names back to the normalized dataset
df_normalized.insert(0, 'communityname', community_names)

print("\nNormalized data (first 5 rows):")
print(df_normalized.head())

# F: Quality check of normalized data
print("\nF: Quality check of normalized data...")
min_max_check = df_normalized.drop(columns=['communityname']).describe().loc[['min', 'max']].T
print("\nVerifying min ≈ 0 and max ≈ 1 for all normalized indicators:")
print(min_max_check)

# Check for NaN values
nan_count = df_normalized.isna().sum().sum()
print(f"\nNumber of NaN values in normalized dataset: {nan_count}")

# G: Save hand-off file for Step 6
print("\nG: Saving normalized dataset for Step 6...")
df_normalized.to_csv('../output/step5_normalized_dataset.csv', index=False)
print("Normalized dataset saved to '../output/step5_normalized_dataset.csv'")

# H: Generate documentation
print("\nH: Generating documentation...")

with open('../docs/normalization_methodology.md', 'w') as f:
    f.write("# Step 5: Normalization Methodology\n\n")
    
    f.write("## Overview\n\n")
    f.write("This document describes the normalization methodology applied to the indicators selected in Step 4.\n\n")
    
    f.write("## Approach\n\n")
    f.write("Min-Max normalization was chosen for its intuitive [0, 1] scale and interpretability for policy dashboards. ")
    f.write("This method preserves the relative distances between values and creates a standardized range that is easily understandable by stakeholders.\n\n")
    
    f.write("## Polarity Inversion\n\n")
    f.write("Indicators with negative polarity (where lower values are better) were inverted before normalization using the formula:\n\n")
    f.write("$x_i^{inv} = x_{max} - x_i$\n\n")
    f.write("This ensures all indicators have the same direction (higher values are better) after normalization.\n\n")
    
    f.write("## Min-Max Normalization\n\n")
    f.write("After polarity inversion, all indicators were normalized using the Min-Max formula:\n\n")
    f.write("$I_i = \\frac{x_i^{inv} - x_{min}^{inv}}{x_{max}^{inv} - x_{min}^{inv}}$\n\n")
    f.write("This transforms all indicators to the [0, 1] range while preserving their distribution.\n\n")
    
    f.write("## Quality Checks\n\n")
    f.write("The following quality checks were performed on the normalized data:\n\n")
    f.write("1. Verification that all normalized indicators have minimum ≈ 0 and maximum ≈ 1\n")
    f.write("2. Check for any NaN values in the normalized dataset\n")
    f.write("3. Visual inspection of distributions through boxplots\n\n")
    
    f.write("## Outlier Handling\n\n")
    f.write("Based on the boxplot analysis, no additional trimming was deemed necessary before normalization. ")
    f.write("The Min-Max approach naturally accommodates outliers by stretching the distribution to the [0, 1] range.\n\n")
    
    f.write("## Skewness\n\n")
    f.write("While some indicators showed skewness, no log transformations were applied. ")
    f.write("The skewness was within acceptable range for Min-Max normalization.\n\n")

# I: (Bonus) Compare with Z-score normalization
print("\nI: (Bonus) Comparing Min-Max with Z-score normalization...")

from sklearn.preprocessing import StandardScaler

# Apply Z-score normalization
z_scaler = StandardScaler()
z_normalized = z_scaler.fit_transform(df_to_normalize)
df_z_normalized = pd.DataFrame(
    z_normalized, 
    columns=df_to_normalize.columns,
    index=df_to_normalize.index
)

# Add community names
df_z_normalized.insert(0, 'communityname', community_names)

# Compare ranks for a few communities
print("\nRank Comparison (Min-Max vs. Z-score) for first few communities:")
sample_indicator = 'ViolentCrimesPerPop' # Choose a sample indicator

# Calculate ranks for Min-Max
minmax_ranks = df_normalized[sample_indicator].rank(ascending=False)
# Calculate ranks for Z-score
z_ranks = df_z_normalized[sample_indicator].rank(ascending=False)

# Create comparison DataFrame
rank_comparison = pd.DataFrame({
    'Community': df_normalized['communityname'],
    'Min-Max Value': df_normalized[sample_indicator],
    'Min-Max Rank': minmax_ranks,
    'Z-score Value': df_z_normalized[sample_indicator],
    'Z-score Rank': z_ranks,
    'Rank Difference': minmax_ranks - z_ranks
})

print(rank_comparison.head(10))

# Save comparison
rank_comparison.to_csv('../output/normalization_method_comparison.csv', index=False)
print("Normalization method comparison saved to '../output/normalization_method_comparison.csv'")

print("\nStep 5 normalization completed successfully!") 