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
    
    