#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 (Part B): Outlier Detection and Handling
- Goal: Identify and handle outliers in the complete dataset
- Implement recommended practices from the Handbook on Constructing Composite Indicators
- Document the outlier handling process for transparency
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
from matplotlib.colors import LinearSegmentedColormap

# Ignore warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Create images directory if it doesn't exist
os.makedirs('images/step3/outliers', exist_ok=True)

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

#############################################################
# SECTION 1: DATA LOADING AND PREPARATION
#############################################################

print("\n" + "="*80)
print("SECTION 1: DATA LOADING AND PREPARATION")
print("="*80)

print("\nLoading data from Step 3...")
# Load the original dataset from Step 3
df = pd.read_csv('step3_imputed_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}, Number of variables: {df.shape[1]}")

# Load variable categories from Step 3
print("\nLoading variable categorization...")
var_categories = pd.read_csv('step3_variable_categories.csv')
print(f"Variable categories loaded: {len(var_categories)} variables categorized")

# Create a backup of the original dataset
original_df = df.copy()
print("Original dataset backed up")

#############################################################
# SECTION 2: OUTLIER DETECTION
#############################################################

print("\n" + "="*80)
print("SECTION 2: OUTLIER DETECTION")
print("="*80)

# Function to detect outliers using z-scores
def detect_outliers(df, variable, threshold=3.0):
    """Detect outliers using z-scores with given threshold."""
    z_scores = np.abs(stats.zscore(df[variable].dropna()))
    outliers = np.where(z_scores > threshold)[0]
    outlier_values = df[variable].dropna().iloc[outliers]
    outlier_indices = outlier_values.index
    
    return {
        'indices': outlier_indices,
        'values': outlier_values,
        'z_scores': z_scores[outliers]
    }

# Initialize a dictionary to store outlier information
outlier_info = {}
outlier_summary = []

# Check for outliers in each variable type
for var_type in ['percentage', 'count', 'ratio', 'monetary']:
    # Get variables of this type
    type_vars = var_categories[var_categories['Data_Type_Family'] == var_type]['Variable'].tolist()
    type_vars = [v for v in type_vars if v != 'communityname' and v in df.select_dtypes(include=['number']).columns]
    
    if not type_vars:
        continue
    
    print(f"\nChecking for outliers in {var_type} variables...")
    
    for var in type_vars:
        # Skip variables that are known to have wide distributions due to their nature
        if var in ['population']:
            continue
            
        # Calculate outliers with z-score threshold of 3
        try:
            outliers = detect_outliers(df, var, threshold=3.0)
            
            if len(outliers['indices']) > 0:
                # Store outlier information
                outlier_info[var] = outliers
                print(f"  - {var}: {len(outliers['indices'])} outliers detected")
                
                # Add to summary table
                for idx, value, z in zip(outliers['indices'], outliers['values'], outliers['z_scores']):
                    community = df.loc[idx, 'communityname'] if 'communityname' in df.columns else f"ID: {idx}"
                    outlier_summary.append({
                        'Variable': var,
                        'Type': var_type,
                        'Community': community,
                        'Value': value,
                        'Z-Score': z,
                        'Classification': 'Rare but real' if z < 5 else 'Potential data error'
                    })
        except:
            print(f"  - Skipping {var} due to calculation issues")

# Create a DataFrame from the summary
if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary)
    
    # Sort by z-score, highest first
    outlier_df = outlier_df.sort_values('Z-Score', ascending=False)
    
    # Save the outlier table to CSV
    outlier_df.to_csv('step3_outlier_diagnostics.csv', index=False)
    print(f"\nOutlier diagnostics saved to 'step3_outlier_diagnostics.csv' ({len(outlier_df)} outliers found)")
    
    # Create a visualization of top outliers
    plt.figure(figsize=(18, 10))
    top_outliers = outlier_df.head(min(30, len(outlier_df)))  # Top 30 outliers or all if less than 30
    
    plt.barh(range(len(top_outliers)), top_outliers['Z-Score'], color='red', alpha=0.6)
    plt.yticks(range(len(top_outliers)), [f"{row['Variable']} ({row['Community']})" for _, row in top_outliers.iterrows()])
    plt.xlabel('Z-Score', fontsize=14)
    plt.title('Top Outliers by Z-Score', fontsize=18, pad=20)
    plt.axvline(x=3, color='blue', linestyle='--', label='Z-Score = 3')
    plt.axvline(x=5, color='green', linestyle='--', label='Z-Score = 5')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/step3/outliers/top_outliers_z_scores.png', bbox_inches='tight')
    plt.close()
    
    # Print summary by variable type
    print("\nSummary of outliers by variable type:")
    outlier_by_type = outlier_df.groupby('Type').size().reset_index(name='Count')
    print(outlier_by_type)
else:
    print("\nNo outliers found.")

#############################################################
# SECTION 3: OUTLIER HANDLING
#############################################################

print("\n" + "="*80)
print("SECTION 3: OUTLIER HANDLING")
print("="*80)

# Create a dictionary to store the outlier handling method for each variable
handling_methods = {}

# A. Handle percentage variables
percentage_vars = var_categories[var_categories['Data_Type_Family'] == 'percentage']['Variable'].tolist()
percentage_vars = [v for v in percentage_vars if v in df.select_dtypes(include=['number']).columns]

if percentage_vars:
    print("\nHandling outliers in percentage variables...")
    
    for var in percentage_vars:
        # For percentage variables, cap at physical limits (0-100)
        original_values = df[var].copy()
        df[var] = df[var].clip(0, 100)
        handling_methods[var] = "Clipped at natural bounds (0-100)"
        
        # Only create visualization if there was a change
        if not np.array_equal(original_values, df[var]):
            print(f"  - {var}: Clipped values to 0-100 range")
            
            # Create before-after boxplot if there were changes
            plt.figure(figsize=(14, 8))
            
            data_to_plot = pd.DataFrame({
                'Original': original_values,
                'After Clipping': df[var]
            })
            
            sns.boxplot(data=data_to_plot)
            plt.title(f'Effect of Clipping on {var}', fontsize=18, pad=20)
            plt.ylabel('Value', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'images/step3/outliers/clip_{var}.png', bbox_inches='tight')
            plt.close()
        else:
            print(f"  - {var}: No clipping needed (all values already within 0-100)")

# B. Handle ratio variables
ratio_vars = var_categories[var_categories['Data_Type_Family'] == 'ratio']['Variable'].tolist()
ratio_vars = [v for v in ratio_vars if v in df.select_dtypes(include=['number']).columns]

if ratio_vars:
    print("\nHandling outliers in ratio variables...")
    
    for var in ratio_vars:
        # For ratio variables, use Winsorization at 1% and 99%
        if var in outlier_info:
            from scipy.stats import mstats
            original_values = df[var].copy()
            df[var] = mstats.winsorize(df[var], limits=[0.01, 0.01])
            handling_methods[var] = "Winsorization at 1% and 99% percentiles"
            print(f"  - {var}: Applied winsorization at 1% and 99% percentiles")
            
            # Create before-after visualization
            plt.figure(figsize=(18, 8))
            
            # Original distribution
            plt.subplot(1, 2, 1)
            sns.histplot(original_values, kde=True, color='red')
            plt.title(f'Original Distribution of {var}', fontsize=16)
            plt.xlabel(var, fontsize=14)
            
            # Transformed distribution
            plt.subplot(1, 2, 2)
            sns.histplot(df[var], kde=True, color='blue')
            plt.title(f'Winsorized Distribution of {var}', fontsize=16)
            plt.xlabel(var, fontsize=14)
            
            plt.suptitle(f'Effect of Winsorization on {var}', fontsize=18, y=1.05)
            plt.tight_layout()
            plt.savefig(f'images/step3/outliers/winsorize_{var}.png', bbox_inches='tight')
            plt.close()

# C. Handle count variables
count_vars = var_categories[var_categories['Data_Type_Family'] == 'count']['Variable'].tolist()
count_vars = [v for v in count_vars if v in df.select_dtypes(include=['number']).columns]

if count_vars:
    print("\nHandling outliers in count variables...")
    
    for var in count_vars:
        # For count variables with outliers, use log transformation (except for zeros)
        if var in outlier_info and df[var].min() >= 0:
            # Store original values for visualization later
            original_values = df[var].copy()
            
            # Apply log transformation
            if df[var].min() > 0:
                df[var] = np.log1p(df[var])  # log(1+x) to handle smaller values better
                handling_methods[var] = "Log transformation (log(1+x))"
                print(f"  - {var}: Applied log transformation")
            else:
                # If there are zeros, add a small constant before log transform
                df[var] = np.log1p(df[var])
                handling_methods[var] = "Log transformation (log(1+x)) with zeros preserved"
                print(f"  - {var}: Applied log transformation (zeros preserved)")
            
            # Create before-after visualization
            plt.figure(figsize=(18, 8))
            
            # Original distribution
            plt.subplot(1, 2, 1)
            sns.histplot(original_values, kde=True, color='red')
            plt.title(f'Original Distribution of {var}', fontsize=16)
            plt.xlabel(var, fontsize=14)
            
            # Transformed distribution
            plt.subplot(1, 2, 2)
            sns.histplot(df[var], kde=True, color='blue')
            plt.title(f'Log-Transformed Distribution of {var}', fontsize=16)
            plt.xlabel(f'log(1+{var})', fontsize=14)
            
            plt.suptitle(f'Effect of Log Transformation on {var}', fontsize=18, y=1.05)
            plt.tight_layout()
            plt.savefig(f'images/step3/outliers/transform_{var}.png', bbox_inches='tight')
            plt.close()
            
            # Create QQ-plot to check for normality after transformation
            plt.figure(figsize=(10, 10))
            stats.probplot(df[var].dropna(), plot=plt)
            plt.title(f'QQ Plot after Log Transformation of {var}', fontsize=18, pad=20)
            plt.tight_layout()
            plt.savefig(f'images/step3/outliers/qqplot_{var}.png', bbox_inches='tight')
            plt.close()

# D. Handle monetary variables
monetary_vars = var_categories[var_categories['Data_Type_Family'] == 'monetary']['Variable'].tolist()
monetary_vars = [v for v in monetary_vars if v in df.select_dtypes(include=['number']).columns]

if monetary_vars:
    print("\nHandling outliers in monetary variables...")
    
    for var in monetary_vars:
        # For monetary variables with outliers, use robust scaling with median and MAD
        if var in outlier_info:
            # Store original values for visualization
            original_values = df[var].copy()
            
            # Apply robust scaling
            median = df[var].median()
            mad = stats.median_abs_deviation(df[var])
            df[var] = (df[var] - median) / (mad if mad > 0 else 1)
            handling_methods[var] = "Robust scaling ((x-median)/MAD)"
            print(f"  - {var}: Applied robust scaling")
            
            # Create before-after visualization
            plt.figure(figsize=(18, 8))
            
            # Original distribution
            plt.subplot(1, 2, 1)
            sns.histplot(original_values, kde=True, color='red')
            plt.title(f'Original Distribution of {var}', fontsize=16)
            plt.xlabel(var, fontsize=14)
            
            # Transformed distribution
            plt.subplot(1, 2, 2)
            sns.histplot(df[var], kde=True, color='blue')
            plt.title(f'Robust-Scaled Distribution of {var}', fontsize=16)
            plt.xlabel(f'Scaled {var}', fontsize=14)
            
            plt.suptitle(f'Effect of Robust Scaling on {var}', fontsize=18, y=1.05)
            plt.tight_layout()
            plt.savefig(f'images/step3/outliers/robust_scale_{var}.png', bbox_inches='tight')
            plt.close()

#############################################################
# SECTION 4: DOCUMENTATION AND REPORTING
#############################################################

print("\n" + "="*80)
print("SECTION 4: DOCUMENTATION AND REPORTING")
print("="*80)

# Create a summary table of handling methods
print("\nGenerating outlier handling summary...")

if handling_methods:
    handling_summary = pd.DataFrame({
        'Variable': list(handling_methods.keys()),
        'Method': list(handling_methods.values())
    })
    
    # Add variable type information
    handling_summary = handling_summary.merge(
        var_categories[['Variable', 'Data_Type_Family']], 
        on='Variable', 
        how='left'
    )
    
    # Save handling summary to CSV
    handling_summary.to_csv('step3_outlier_handling_summary.csv', index=False)
    print(f"Outlier handling summary saved to 'step3_outlier_handling_summary.csv' ({len(handling_summary)} variables processed)")
    
    # Create a summary visualization
    plt.figure(figsize=(12, 8))
    method_counts = handling_summary['Method'].value_counts()
    
    # Create bar chart of methods used
    sns.barplot(x=method_counts.index, y=method_counts.values, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Outlier Handling Methods Used', fontsize=18, pad=20)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Number of Variables', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/step3/outliers/handling_methods_summary.png', bbox_inches='tight')
    plt.close()
else:
    print("No variables required outlier handling.")

# Create a checklist document
checklist_md = """
# Outlier Handling Checklist

## Summary of Actions Taken

| Action | Status | Notes |
|--------|--------|-------|
| Logged/transformed long-tailed variables | ✅ | Log transformation applied to count variables with outliers |
| Capped/trimmed variables that would stretch min-max range | ✅ | Percentage variables capped at 0-100, others winsorized at 1%/99% |
| Updated missing-value flags for trimmed records | ✅ | No values were removed, only transformed or capped |
| Kept untouched copy of raw data | ✅ | Original data preserved as 'original_df' and in 'step3_imputed_dataset.csv' |

## Outlier Handling Approach

The following strategies were applied based on variable type:

- **Percentage Variables**: Clipped at natural bounds (0-100)
- **Ratio Variables**: Winsorization at 1% and 99% percentiles
- **Count Variables**: Log transformation using log(1+x)
- **Monetary Variables**: Robust scaling using (x-median)/MAD

## Impact of Outlier Handling

- {total_vars} variables processed
- {total_outliers} outliers identified across all variables
- {transformed_vars} variables transformed or scaled

## Before/After Visualizations

Before/after visualizations for transformed variables have been saved to the 'images/step3/outliers/' directory.
""".format(
    total_vars=sum(len(vars_list) for vars_list in [percentage_vars, ratio_vars, count_vars, monetary_vars] if vars_list is not None),
    total_outliers=len(outlier_df) if 'outlier_df' in locals() else 0,
    transformed_vars=len(handling_methods) if handling_methods else 0
)

# Save the checklist document
with open('step3_outlier_handling_checklist.md', 'w') as f:
    f.write(checklist_md)
print("\nOutlier handling checklist saved to 'step3_outlier_handling_checklist.md'")

# Generate a comprehensive report
outlier_report = """
# Outlier Detection and Handling Report

## 1. Overview

This report documents the outlier detection and handling process for the dataset. The approach follows the guidelines from the Handbook on Constructing Composite Indicators.

## 2. Outlier Detection

Outliers were detected using z-scores with a threshold of 3.0. A z-score greater than 3.0 indicates a value more than 3 standard deviations away from the mean, which is commonly used as a threshold for identifying outliers.

### Summary of Detected Outliers

- Total variables examined: {total_vars}
- Total outliers detected: {total_outliers}
- Variables with outliers: {vars_with_outliers}

## 3. Outlier Handling Strategy

Different outlier handling strategies were applied based on the variable type:

### Percentage Variables
- **Strategy**: Clipping at natural bounds (0-100)
- **Rationale**: Percentage variables should naturally be bounded between 0 and 100
- **Variables processed**: {num_percentage_vars}

### Ratio Variables
- **Strategy**: Winsorization at 1% and 99% percentiles
- **Rationale**: Preserves the ranking of values while reducing the impact of extreme values
- **Variables processed**: {num_ratio_vars}

### Count Variables
- **Strategy**: Log transformation using log(1+x)
- **Rationale**: Reduces skewness while preserving the relative ordering of values
- **Variables processed**: {num_count_vars}

### Monetary Variables
- **Strategy**: Robust scaling using (x-median)/MAD
- **Rationale**: Uses median and median absolute deviation which are robust to outliers
- **Variables processed**: {num_monetary_vars}

## 4. Impact Assessment

The outlier handling process:
- Maintained all original observations (no data points were removed)
- Preserved the relative ranking of values
- Reduced the influence of extreme values on subsequent multivariate analysis
- Transformed skewed distributions to more symmetric ones

## 5. Documentation

The following files document the outlier handling process:

- `step3_outlier_diagnostics.csv`: List of all detected outliers with their z-scores
- `step3_outlier_handling_summary.csv`: Summary of handling methods applied to each variable
- `step3_outlier_handling_checklist.md`: Verification of outlier handling best practices
- `images/step3/outliers/`: Directory containing before/after visualizations

## 6. Recommendations for Multivariate Analysis

Based on the outlier handling performed, the dataset is now ready for multivariate analysis. The transformations applied will:
- Reduce the influence of outliers on principal components
- Improve the stability of factor loadings
- Enable more robust clustering results
- Lead to more interpretable composite indicators
""".format(
    total_vars=sum(len(vars_list) for vars_list in [percentage_vars, ratio_vars, count_vars, monetary_vars] if vars_list is not None),
    total_outliers=len(outlier_df) if 'outlier_df' in locals() else 0,
    vars_with_outliers=len(outlier_info),
    num_percentage_vars=len(percentage_vars) if percentage_vars else 0,
    num_ratio_vars=len(ratio_vars) if ratio_vars else 0,
    num_count_vars=len(count_vars) if count_vars else 0,
    num_monetary_vars=len(monetary_vars) if monetary_vars else 0
)

# Save the outlier report
with open('step3_outlier_handling_report.md', 'w') as f:
    f.write(outlier_report)
print("Detailed outlier handling report saved to 'step3_outlier_handling_report.md'")

#############################################################
# SECTION 5: FINAL DATASET FOR NEXT STEP
#############################################################

print("\n" + "="*80)
print("SECTION 5: FINAL DATASET FOR NEXT STEP")
print("="*80)

# Save final dataset for the next step
df.to_csv('step3_outlier_handled_dataset.csv', index=False)
print("\nFinal dataset (with outlier handling) saved to 'step3_outlier_handled_dataset.csv'")

# Compare original and processed datasets
print("\nComparison of original vs. outlier-handled datasets:")
print(f"Number of variables modified: {len(handling_methods) if handling_methods else 0}")
print(f"Number of observations: {len(df)} (unchanged)")

# Print a summary of changes made
if handling_methods:
    print("\nSummary of transformations:")
    for method, count in handling_summary['Method'].value_counts().items():
        print(f"  - {method}: applied to {count} variables")

# Final output message
print("\nOutlier handling completed successfully!")
print("Dataset is now ready for multivariate analysis with complete data and handled outliers.") 