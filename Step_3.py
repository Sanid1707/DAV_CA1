#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Imputation of Missing Data
- Goal: Produce one complete, logically consistent tabular data set
- Record how every missing value was handled (transparency)
- Quantify impact of imputation on data reliability
"""

#############################################################
# SECTION 1: INITIALIZATION AND SETUP
# - Import libraries
# - Configure visualization settings
# - Set up environment
#############################################################

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
import missingno as msno
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
from sklearn.linear_model import BayesianRidge
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Ignore FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Create images directory if it doesn't exist
os.makedirs('images/step3', exist_ok=True)

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
# SECTION 2: DATA LOADING AND PREPROCESSING
# - Load data from previous step
# - Analyze missing values
# - Categorize variables by data type
#############################################################

print("\n" + "="*80)
print("SECTION 2: DATA LOADING AND PREPROCESSING")
print("="*80)

# Load the "clean-stage" file produced after variable-specific QC
print("\nLoading the dataset from Step 2...")
df = pd.read_csv('step2_final_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}, Number of variables: {df.shape[1]}")

# Save a copy as the starting point for imputation
df.to_csv('03_preimp_raw.csv', index=False)
print("Dataset saved as '03_preimp_raw.csv'")

# Create a "missing-values map"
print("\nCreating missing values map...")

# Function to calculate missing data statistics
def missing_data_summary(df):
    # Count of missing values
    missing_count = df.isnull().sum()
    
    # Percentage of missing values
    missing_percent = (missing_count / len(df)) * 100
    
    # Create a summary DataFrame
    missing_summary = pd.DataFrame({
        'Variable': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_percent.values
    })
    
    # Sort by percentage of missing values (descending)
    missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)
    
    return missing_summary

# Get missing data summary
missing_summary = missing_data_summary(df)

# Display variables with missing values
missing_with_nulls = missing_summary[missing_summary['Missing_Count'] > 0]
print("\nVariables with missing values:")
print(missing_with_nulls)

# Save missing values summary to CSV
missing_summary.to_csv('step3_missing_values_summary.csv', index=False)
print("Missing values summary saved to 'step3_missing_values_summary.csv'")

# A3: Tag each column with its family (data type category)
print("\nA3: Tagging columns with their data type families...")

# Define a function to categorize variables
def categorize_variables(df):
    # Initialize metadata dictionary
    metadata = {}
    
    # Define explicit mappings for variables that need correction
    explicit_mappings = {
        'communityname': 'identifier',
        'population': 'count',
        'racepctblack': 'percentage',  # Corrected from count to percentage
        'racePctWhite': 'percentage',
        'racePctAsian': 'percentage',
        'racePctHisp': 'percentage',
        'pctUrban': 'percentage',
        'medIncome': 'monetary',
        'pctWSocSec': 'percentage',
        'pctWPubAsst': 'percentage',
        'PctPopUnderPov': 'percentage',
        'PctUnemployed': 'percentage',
        'PersPerFam': 'ratio',
        'PctFam2Par': 'percentage',
        'PctImmigRec5': 'percentage',
        'PctRecentImmig': 'percentage',
        'PctNotSpeakEnglWell': 'percentage',
        'PctLargHouseFam': 'percentage',
        'PersPerRentOccHous': 'ratio',
        'PctPersDenseHous': 'percentage',
        'PctHousOccup': 'percentage',
        'PctVacantBoarded': 'percentage',
        'PctHousNoPhone': 'percentage',
        'PctWOFullPlumb': 'percentage',
        'OwnOccQrange': 'count',
        'RentQrange': 'count',
        'NumInShelters': 'count',
        'PctSameHouse85': 'percentage',
        'PctUsePubTrans': 'percentage',
        # Crime counts
        'murders': 'count',
        'rapes': 'count',
        'robberies': 'count',
        'assaults': 'count',
        'burglaries': 'count',
        'larcenies': 'count',
        'autoTheft': 'count',
        'arsons': 'count',
        # Crime rates (per population)
        'murdPerPop': 'ratio',
        'rapesPerPop': 'ratio',
        'robbbPerPop': 'ratio',
        'assaultPerPop': 'ratio',
        'burglPerPop': 'ratio',
        'larcPerPop': 'ratio',
        'autoTheftPerPop': 'ratio',
        'arsonsPerPop': 'ratio',
        'ViolentCrimesPerPop': 'ratio',
        'nonViolPerPop': 'ratio'
    }
    
    for col in df.columns:
        # First check if there's an explicit mapping
        if col in explicit_mappings:
            metadata[col] = explicit_mappings[col]
            continue
            
        # If no explicit mapping, apply logic-based categorization
        # Check if the column contains percentage data (based on name or range)
        if col.startswith('pct') or col.startswith('Pct') or 'Pct' in col:
            metadata[col] = 'percentage'
            
            # Verify percentage values are in proper range
            if df[col].dropna().max() > 100 or df[col].dropna().min() < 0:
                print(f"Warning: Variable {col} has values outside the 0-100 range but is classified as percentage")
        
        # Check if the column might be a monetary value
        elif col.startswith('med') or 'Income' in col or 'Cost' in col:
            metadata[col] = 'monetary'
            
        # Check if column contains count data (often integer values)
        elif df[col].dtype == 'int64' or (df[col].dtype == 'float64' and df[col].dropna().apply(lambda x: x.is_integer()).all()):
            # Check if the values are small enough to be ratios
            if df[col].max() <= 100 and df[col].min() >= 0:
                metadata[col] = 'ratio'
            else:
                metadata[col] = 'count'
        
        # If still not categorized, check name patterns
        elif 'PerPop' in col or 'Per' in col or 'Ratio' in col:
            metadata[col] = 'ratio'
        
        # Default to count if nothing else matches
        else:
            metadata[col] = 'count'
    
    return pd.DataFrame({
        'Variable': list(metadata.keys()),
        'Data_Type_Family': list(metadata.values())
    })

# Generate variable categorization
var_categories = categorize_variables(df)

# Display variable categorization summary
print("\nVariable categorization summary:")
print(var_categories['Data_Type_Family'].value_counts())

# Save variable categorization to CSV
var_categories.to_csv('step3_variable_categories.csv', index=False)
print("Variable categories saved to 'step3_variable_categories.csv'")

#############################################################
# SECTION 3: DATA COMPLETENESS ANALYSIS
# - Check for missing values
# - Create visualizations of data completeness
# - Generate variable type distribution visualizations
#############################################################

print("\n" + "="*80)
print("SECTION 3: DATA COMPLETENESS ANALYSIS")
print("="*80)

# Check if there are any missing values
if missing_with_nulls.empty:
    print("\nNo missing values found in the dataset. Data is already complete!")
    
    # Create directory for step3 images if it doesn't exist
    os.makedirs('images/step3', exist_ok=True)
    
    # DATA COMPLETENESS VISUALIZATIONS
    print("\nCreating data completeness visualizations...")
    
    # Create a visualization to show data completeness
    plt.figure(figsize=(20, 12))
    plt.title('Data Completeness Visualization - No Missing Values', fontsize=22, pad=20)
    ax = sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    ax.set_xlabel("Variables", fontsize=18, labelpad=15)
    ax.set_ylabel("Observations", fontsize=18, labelpad=15)
    plt.tight_layout()
    plt.savefig('images/step3/01_data_completeness.png', bbox_inches='tight')
    plt.close()
    
    # Create matrix plot showing completeness
    plt.figure(figsize=(20, 12))
    msno.matrix(df, fontsize=12, color=(0.4, 0.21, 0.58))
    plt.title('Data Completeness Matrix - No Missing Values', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/step3/02_data_completeness_matrix.png', bbox_inches='tight')
    plt.close()
    
    # Add a dendrogram plot to show variable relationships
    plt.figure(figsize=(20, 12))
    msno.dendrogram(df, orientation='top', figsize=(20, 12), fontsize=12)
    plt.title('Dendrogram of Variable Relationships - Complete Data', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/step3/03_variable_relationships_dendrogram.png', bbox_inches='tight')
    plt.close()
    
    #############################################################
    # SECTION 4: DATA VISUALIZATION AND ANALYSIS
    # - Create variable type visualizations
    # - Analyze distributions and ranges
    # - Generate correlation heatmaps and relationship plots
    #############################################################
    
    print("\n" + "="*80)
    print("SECTION 4: DATA VISUALIZATION AND ANALYSIS")
    print("="*80)
    
    # VARIABLE TYPE DISTRIBUTION VISUALIZATIONS
    print("\nCreating variable type distribution visualizations...")
    
    # Create a visual summary of variable types - Bar Chart
    plt.figure(figsize=(14, 10))
    type_counts = var_categories['Data_Type_Family'].value_counts()
    ax = sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis')
    plt.title('Distribution of Variable Types', fontsize=22, pad=20)
    plt.xlabel('Variable Type', fontsize=18, labelpad=15)
    plt.ylabel('Count', fontsize=18, labelpad=15)
    
    # Add count labels on top of bars
    for i, v in enumerate(type_counts.values):
        ax.text(i, v + 0.25, str(v), ha='center', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/step3/04_variable_types_distribution.png', bbox_inches='tight')
    plt.close()
    
    # Create a pie chart of variable types
    plt.figure(figsize=(12, 12))
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
            shadow=False, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 14, 'fontweight': 'bold'},
            colors=sns.color_palette('viridis', n_colors=len(type_counts)))
    plt.axis('equal')
    plt.title('Distribution of Variable Types', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/step3/05_variable_types_pie.png', bbox_inches='tight')
    plt.close()
    
    # 3. DETAILED VARIABLE TYPE ANALYSIS
    # Create a heatmap showing variable type distribution
    # Convert variable types to numeric for heatmap visualization
    type_mapping = {'percentage': 0, 'count': 1, 'ratio': 2, 'monetary': 3, 'identifier': 4}
    var_categories['type_code'] = var_categories['Data_Type_Family'].map(type_mapping)
    
    # Create a matrix form for visualization
    var_matrix = np.zeros((5, 10))  # 5 types, showing 10 variables per row
    for i, family in enumerate(['percentage', 'count', 'ratio', 'monetary', 'identifier']):
        vars_of_type = var_categories[var_categories['Data_Type_Family'] == family]['Variable'].values
        for j, var in enumerate(vars_of_type[:10]):  # Show up to 10 variables for each type
            if j < 10:
                var_matrix[i, j] = 1
    
    # Create a custom colormap
    colors = ['lightgray', '#4c02a1']  # Empty to filled
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=2)
    
    # Create the heatmap
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(var_matrix, cmap=cmap, cbar=False, 
                      linewidths=1, linecolor='white', 
                      annot=False, square=True)
    
    # Add custom labels
    ax.set_xticks(np.arange(10) + 0.5)
    ax.set_xticklabels([f'Var {i+1}' for i in range(10)], fontsize=12)
    
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_yticklabels(['Percentage', 'Count', 'Ratio', 'Monetary', 'Identifier'], fontsize=14)
    
    # Add legend
    handles = [
        mpatches.Patch(color='#4c02a1', label=f'Percentage: {len(var_categories[var_categories["Data_Type_Family"] == "percentage"])}'),
        mpatches.Patch(color='#5d177e', label=f'Count: {len(var_categories[var_categories["Data_Type_Family"] == "count"])}'),
        mpatches.Patch(color='#7b2293', label=f'Ratio: {len(var_categories[var_categories["Data_Type_Family"] == "ratio"])}'),
        mpatches.Patch(color='#982cae', label=f'Monetary: {len(var_categories[var_categories["Data_Type_Family"] == "monetary"])}'),
        mpatches.Patch(color='#b936cb', label=f'Identifier: {len(var_categories[var_categories["Data_Type_Family"] == "identifier"])}')
    ]
    plt.legend(handles=handles, loc='upper right', fontsize=14, frameon=True)
    
    plt.title('Variable Type Distribution Map', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/step3/06_variable_type_map.png', bbox_inches='tight')
    plt.close()
    
    # SUMMARY STATISTICS BY VARIABLE TYPE
    print("\nCreating summary statistics visualizations...")
    
    # Create summary statistics grid with individual visualizations for each type
    for i, var_type in enumerate(['percentage', 'count', 'ratio', 'monetary']):
        # Get variables of this type
        type_vars = var_categories[var_categories['Data_Type_Family'] == var_type]['Variable'].tolist()
        
        if not type_vars:
            continue
            
        # Filter out identifier column
        type_vars = [v for v in type_vars if v != 'communityname']
        
        # Take a sample of variables if there are too many
        sample_vars = type_vars[:min(6, len(type_vars))]
        
        # Create summary statistics plot
        plt.figure(figsize=(18, 10))
        stats_df = df[sample_vars].describe().transpose()
        stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        
        ax = stats_df.plot(kind='bar', figsize=(18, 10), colormap='viridis')
        ax.set_title(f'Summary Statistics for {var_type.title()} Variables', fontsize=22, pad=20)
        ax.set_ylabel('Value', fontsize=18, labelpad=15)
        ax.set_xlabel('Variable', fontsize=18, labelpad=15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add a legend with clearer labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Statistic', fontsize=14, title_fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'images/step3/07_summary_stats_{var_type}.png', bbox_inches='tight')
        plt.close()
        
        # Create boxplots for each variable type separately
        plt.figure(figsize=(18, 10))
        boxplot = df[sample_vars].boxplot(vert=False, patch_artist=True, 
                                         boxprops=dict(facecolor='#440154', alpha=0.8),
                                         medianprops=dict(color='white', linewidth=2),
                                         flierprops=dict(marker='o', markerfacecolor='#fde725', markersize=8),
                                         return_type='dict')
        plt.title(f'Boxplots for {var_type.title()} Variables', fontsize=22, pad=20)
        plt.xlabel('Value', fontsize=18, labelpad=15)
        plt.tight_layout()
        plt.savefig(f'images/step3/08_boxplots_{var_type}.png', bbox_inches='tight')
        plt.close()
    
    # 5. VARIABLE RANGE ANALYSIS
    # Create a heatmap showing ranges by variable type
    range_data = pd.DataFrame()
    for var_type in ['percentage', 'count', 'ratio', 'monetary']:
        type_vars = var_categories[var_categories['Data_Type_Family'] == var_type]['Variable'].tolist()
        type_vars = [v for v in type_vars if v != 'communityname' and v in df.columns]
        
        if not type_vars:
            continue
            
        # Calculate ranges for each variable
        for var in type_vars:
            range_data.loc[var, 'type'] = var_type
            range_data.loc[var, 'min'] = df[var].min()
            range_data.loc[var, 'max'] = df[var].max()
            range_data.loc[var, 'range'] = df[var].max() - df[var].min()
            range_data.loc[var, 'mean'] = df[var].mean()
            range_data.loc[var, 'std'] = df[var].std()
    
    # Create range visualization
    if not range_data.empty:
        pivot_data = range_data.pivot_table(index='type', values=['min', 'max', 'range', 'mean', 'std'], aggfunc='mean')
        
        plt.figure(figsize=(18, 10))
        ax = sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', linewidths=0.5, linecolor='white')
        ax.set_title('Average Statistics by Variable Type', fontsize=22, pad=20)
        ax.set_xlabel('Statistic', fontsize=18, labelpad=15)
        ax.set_ylabel('Variable Type', fontsize=18, labelpad=15)
        plt.tight_layout()
        plt.savefig('images/step3/09_variable_type_ranges.png', bbox_inches='tight')
        plt.close()
    
    # 6. DISTRIBUTION ANALYSIS BY VARIABLE TYPE
    # Create individual histograms for each variable type
    for var_type in ['percentage', 'count', 'ratio', 'monetary']:
        type_vars = var_categories[var_categories['Data_Type_Family'] == var_type]['Variable'].tolist()
        type_vars = [v for v in type_vars if v != 'communityname']
        
        if not type_vars:
            continue
            
        # Take a sample of variables if there are too many
        sample_vars = type_vars[:min(9, len(type_vars))]
        
        # Create distribution plots
        plt.figure(figsize=(20, 15))
        for i, var in enumerate(sample_vars, 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[var], kde=True, color='#3b0f70')
            plt.plot([], [], color='#fde725', linewidth=2, label='KDE')  # Add a dummy line for legend
            plt.title(f'Distribution of {var}', fontsize=16)
            plt.xlabel(var, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.legend()
        
        plt.suptitle(f'Distributions of {var_type.title()} Variables', fontsize=22, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'images/step3/10_distributions_{var_type}.png', bbox_inches='tight')
        plt.close()
    
    # 7. PERCENTAGE VARIABLE VALIDATION
    # Check that percentage variables are all within 0-100 range
    percentage_vars = var_categories[var_categories['Data_Type_Family'] == 'percentage']['Variable'].tolist()
    percent_in_range = True
    out_of_range_vars = []
    
    for var in percentage_vars:
        if df[var].min() < 0 or df[var].max() > 100:
            percent_in_range = False
            out_of_range_vars.append(var)
    
    # Create a percentage range validation visualization
    plt.figure(figsize=(18, 10))
    
    # Plot a horizontal line at 0 and 100 to show valid range
    plt.axhline(y=0, color='#fde725', linestyle='-', linewidth=2, alpha=0.5)
    plt.axhline(y=100, color='#fde725', linestyle='-', linewidth=2, alpha=0.5)
    
    # Plot min and max for each percentage variable
    percentage_range_data = pd.DataFrame(index=percentage_vars)
    for var in percentage_vars:
        percentage_range_data.loc[var, 'min'] = df[var].min()
        percentage_range_data.loc[var, 'max'] = df[var].max()
    
    # Sort by min value
    percentage_range_data = percentage_range_data.sort_values('min')
    
    # Plot min and max as error bars
    plt.errorbar(x=np.arange(len(percentage_range_data)), 
                 y=percentage_range_data['min'] + (percentage_range_data['max'] - percentage_range_data['min'])/2,
                 yerr=(percentage_range_data['max'] - percentage_range_data['min'])/2, 
                 fmt='o', color='#440154', ecolor='#440154', elinewidth=1.5, capsize=5, markersize=8)
    
    plt.xticks(np.arange(len(percentage_range_data)), percentage_range_data.index, rotation=90)
    plt.title('Percentage Variables Range Validation', fontsize=22, pad=20)
    plt.ylabel('Value Range (0-100 is valid)', fontsize=18, labelpad=15)
    plt.xlabel('Percentage Variables', fontsize=18, labelpad=15)
    
    # Shade the valid region
    plt.axhspan(0, 100, alpha=0.1, color='green')
    
    # Add text explaining the plot
    if percent_in_range:
        plt.figtext(0.5, 0.01, 'All percentage variables are within the valid 0-100 range', 
                   ha='center', fontsize=14, bbox=dict(facecolor='#fde725', alpha=0.1))
    else:
        plt.figtext(0.5, 0.01, f'Warning: {len(out_of_range_vars)} percentage variables are outside the valid 0-100 range', 
                   ha='center', fontsize=14, bbox=dict(facecolor='red', alpha=0.1))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('images/step3/11_percentage_range_validation.png', bbox_inches='tight')
    plt.close()
    
    if not percent_in_range:
        # Create visualizations for out-of-range percentage variables
        plt.figure(figsize=(18, 12))
        for i, var in enumerate(out_of_range_vars[:min(len(out_of_range_vars), 4)], 1):
            plt.subplot(2, 2, i)
            sns.histplot(df[var], kde=True, color='red', alpha=0.6)
            plt.axvline(x=0, color='green', linestyle='--', linewidth=2)
            plt.axvline(x=100, color='green', linestyle='--', linewidth=2)
            plt.title(f'Distribution of {var} (Outside 0-100 range)', fontsize=16)
            plt.xlabel(var, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
        
        plt.suptitle('Out-of-Range Percentage Variables', fontsize=22, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('images/step3/12_out_of_range_percentages.png', bbox_inches='tight')
        plt.close()
    
    # 8. CRIME RATE ANALYSIS
    # Create crime rate comparisons
    crime_rate_vars = [col for col in df.columns if 'PerPop' in col]
    if crime_rate_vars:
        # Create boxplot of crime rates
        plt.figure(figsize=(18, 12))
        box_colors = [plt.cm.viridis(i/float(len(crime_rate_vars))) for i in range(len(crime_rate_vars))]
        
        box = df[crime_rate_vars].boxplot(vert=False, patch_artist=True, 
                                         boxprops=dict(alpha=0.8),
                                         medianprops=dict(color='white', linewidth=2),
                                         flierprops=dict(marker='o', markersize=8),
                                         return_type='dict')
        
        # Color the boxes
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
            
        plt.title('Distribution of Crime Rates', fontsize=22, pad=20)
        plt.xlabel('Rate per Population', fontsize=18, labelpad=15)
        plt.tight_layout()
        plt.savefig('images/step3/13_crime_rates_comparison.png', bbox_inches='tight')
        plt.close()
        
        # Crime rate correlation heatmap
        plt.figure(figsize=(18, 16))
        crime_corr = df[crime_rate_vars].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(crime_corr, dtype=bool))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(crime_corr, annot=True, cmap=cmap, fmt='.2f', 
                   linewidths=0.5, vmin=-1, vmax=1, square=True, mask=mask)
        plt.title('Correlation Between Different Crime Rates', fontsize=22, pad=20)
        plt.tight_layout()
        plt.savefig('images/step3/14_crime_rates_correlation.png', bbox_inches='tight')
        plt.close()
        
        # Create a pairplot for crime rates (top 4 variables)
        top_crimes = crime_rate_vars[:min(4, len(crime_rate_vars))]
        plt.figure(figsize=(20, 20))
        
        # Set up the grid for plots
        n = len(top_crimes)
        for i, var1 in enumerate(top_crimes):
            for j, var2 in enumerate(top_crimes):
                # Create subplot
                plt.subplot(n, n, i*n + j + 1)
                
                if i == j:  # Diagonal: histograms
                    sns.histplot(df[var1], kde=True, color='#440154')
                    plt.title(var1.replace('PerPop', ''), fontsize=14)
                else:  # Off-diagonal: scatter plots
                    plt.scatter(df[var2], df[var1], alpha=0.5, s=20, color='#440154')
                    plt.xlabel(var2.replace('PerPop', ''), fontsize=12)
                    plt.ylabel(var1.replace('PerPop', ''), fontsize=12)
                    
                    # Add correlation coefficient
                    corr = df[var1].corr(df[var2])
                    plt.annotate(f'r = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                                ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
                                                                          facecolor='white', alpha=0.7))
        
        plt.suptitle('Relationships Between Major Crime Rates', fontsize=22, y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('images/step3/15_crime_rates_relationships.png', bbox_inches='tight')
        plt.close()
    
    # 9. CORRELATION ANALYSIS
    # Add correlation heatmap between variables
    plt.figure(figsize=(24, 20))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=False, square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of All Variables', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/step3/16_correlation_heatmap_all.png', bbox_inches='tight')
    plt.close()
    
    # Create a focused correlation heatmap for top correlated variables
    # Get the most correlated pairs
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append((corr.index[i], corr.columns[j], abs(corr.iloc[i, j])))
    
    # Sort by absolute correlation and take top 20
    top_corrs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:20]
    
    # Create a smaller correlation matrix with just these variables
    top_vars = list(set([pair[0] for pair in top_corrs] + [pair[1] for pair in top_corrs]))
    top_corr = numeric_df[top_vars].corr()
    
    plt.figure(figsize=(18, 15))
    sns.heatmap(top_corr, annot=True, cmap=cmap, vmax=1, vmin=-1, center=0,
            fmt='.2f', square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Top Correlated Variables', fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig('images/step3/17_correlation_heatmap_top.png', bbox_inches='tight')
    plt.close()
    
    # GEOGRAPHIC DISTRIBUTION
    print("\nCreating geographic analysis visualizations...")
    
    # Create geographical distribution if communityname is available
    if 'communityname' in df.columns:
        # Extract state from communityname (assuming format like "community, ST")
        if df['communityname'].str.contains(',').any():
            df['state'] = df['communityname'].str.split(',').str[1].str.strip().str[:2]
            
            # Plot count by state
            plt.figure(figsize=(20, 12))
            state_counts = df['state'].value_counts().sort_values(ascending=False)
            ax = sns.barplot(x=state_counts.index[:20], y=state_counts.values[:20], palette='viridis')
            
            # Add count labels on top of bars
            for i, v in enumerate(state_counts.values[:20]):
                ax.text(i, v + 5, str(v), ha='center', fontsize=12, fontweight='bold')
                
            plt.title('Number of Communities by State (Top 20)', fontsize=22, pad=20)
            plt.xticks(rotation=45)
            plt.xlabel('State', fontsize=18, labelpad=15)
            plt.ylabel('Number of Communities', fontsize=18, labelpad=15)
            plt.tight_layout()
            plt.savefig('images/step3/18_communities_by_state.png', bbox_inches='tight')
            plt.close()
            
            # Create boxplots of violent crime by state
            if 'ViolentCrimesPerPop' in df.columns:
                plt.figure(figsize=(20, 12))
                top_states = state_counts.index[:12]  # Top 12 states by count
                top_states_df = df[df['state'].isin(top_states)]
                
                # Sort states by median violent crime rate
                state_crime_medians = top_states_df.groupby('state')['ViolentCrimesPerPop'].median().sort_values(ascending=False)
                
                # Create boxplot with states in order of median crime rate
                ax = sns.boxplot(x='state', y='ViolentCrimesPerPop', data=top_states_df, 
                             order=state_crime_medians.index, palette='viridis')
                
                plt.title('Violent Crime Rates by State (Top 12 States by Count)', fontsize=22, pad=20)
                plt.xlabel('State', fontsize=18, labelpad=15)
                plt.ylabel('Violent Crimes Per Population', fontsize=18, labelpad=15)
                
                # Add median labels
                for i, state in enumerate(state_crime_medians.index):
                    median = state_crime_medians[state]
                    ax.text(i, median, f'{median:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                plt.tight_layout()
                plt.savefig('images/step3/19_violent_crime_by_state.png', bbox_inches='tight')
                plt.close()
                
                # Create a heatmap of crime rates by state
                pivot_crime = pd.DataFrame()
                crime_cols = [col for col in df.columns if 'PerPop' in col][:5]  # Take top 5 crime rate variables
                
                for crime in crime_cols:
                    pivot_crime[crime] = top_states_df.groupby('state')[crime].mean()
                
                plt.figure(figsize=(16, 12))
                sns.heatmap(pivot_crime, annot=True, fmt='.4f', cmap='viridis', linewidths=0.5)
                plt.title('Average Crime Rates by State (Top 12 States)', fontsize=22, pad=20)
                plt.tight_layout()
                plt.savefig('images/step3/20_crime_heatmap_by_state.png', bbox_inches='tight')
                plt.close()
