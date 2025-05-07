# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#############################################################
# SECTION 1: SETUP AND INITIALIZATION
#############################################################
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from statsmodels.stats.outliers_influence import variance_inflation_factor
import networkx as nx
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Set style for the plots - Using a more modern Seaborn style
plt.style.use('seaborn-v0_8-whitegrid') # Cleaner style
sns.set_palette('viridis') # Perceptually uniform categorical palette
plt.rcParams['figure.figsize'] = (16, 10) # Default figure size
plt.rcParams['font.size'] = 12 # Default font size
plt.rcParams['axes.titlesize'] = 16 # Title font size
plt.rcParams['axes.labelsize'] = 14 # Axis label font size
plt.rcParams['xtick.labelsize'] = 12 # X-tick label size
plt.rcParams['ytick.labelsize'] = 12 # Y-tick label size
plt.rcParams['legend.fontsize'] = 12 # Legend font size
plt.rcParams['figure.dpi'] = 100 # Default DPI for screen
plt.rcParams['savefig.dpi'] = 300 # Higher DPI for saved figures

# Try different encodings to load the dataset
try:
    # Try with Latin-1 encoding
    df = pd.read_csv('crimedata.csv', encoding='latin1')
except:
    try:
        # Try with ISO-8859-1 encoding
        df = pd.read_csv('crimedata.csv', encoding='ISO-8859-1')
    except:
        try:
            # Try with Windows-1252 encoding
            df = pd.read_csv('crimedata.csv', encoding='cp1252')
        except:
            # Fallback to detect encoding
            import chardet
            with open('crimedata.csv', 'rb') as file:
                result = chardet.detect(file.read())
            df = pd.read_csv('crimedata.csv', encoding=result['encoding'])

#############################################################
# SECTION 2: DATA CLEANING FUNCTIONS
#############################################################

# Fix column names with encoding issues
def fix_column_names(df):
    """Fix encoding issues in column names."""
    renamed_columns = {}
    for col in df.columns:
        if col.startswith('Ê'):
            new_name = col[1:]  # Remove the first character
            renamed_columns[col] = new_name
            print(f"  - Renamed '{col}' to '{new_name}'")
    
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
    return df

# Replace question marks with NaN and log the counts
def handle_question_marks(df):
    """Replace question marks in the dataset with NaN and log the counts."""
    df_clean = df.copy()
    question_mark_counts = {}
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Count question marks
            q_marks = (df_clean[col] == '?').sum()
            if q_marks > 0:
                question_mark_counts[col] = q_marks
                print(f"  - Replaced {q_marks} question marks in column '{col}'")
                df_clean.loc[df_clean[col] == '?', col] = np.nan
                
    # Save question mark counts to CSV
    pd.DataFrame({
        'Variable': list(question_mark_counts.keys()),
        'Question Mark Count': list(question_mark_counts.values())
    }).to_csv('step2_question_mark_counts.csv', index=False)
    
    return df_clean

# Convert columns to numeric where possible
def convert_to_numeric(df):
    """Attempt to convert columns to numeric types."""
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if col not in ['communityname', 'state', 'countyCode', 'communityCode', 'fold']:
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
            except:
                pass
    return df_numeric

# Create a function to calculate missing percentage
def missing_percentage(df):
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Variable': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing %': missing_percent.values
    })
    return missing_df.sort_values('Missing %', ascending=False)

# Handle implausible zeros - ONLY for variables where zeros are genuinely implausible
def handle_implausible_zeros(df):
    """Replace zeros with NaN only in variables where zeros are truly implausible."""
    df_clean = df.copy()
    
    # Define columns where zeros are implausible (excluding crime rates)
    zero_to_nan_cols = [
        'medIncome', 'PopDens', 'MedYrHousBuilt', 'NumPolice', 'PolicPerPop'
    ]
    
    for col in zero_to_nan_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            zeros = (df_clean[col] == 0).sum()
            if zeros > 0:
                print(f"  - Replaced {zeros} zeros with NaN in column '{col}' (zeros implausible)")
                df_clean.loc[df_clean[col] == 0, col] = np.nan
    
    return df_clean

# Function to check for constant columns
def find_constant_columns(df):
    """Find columns with zero variance."""
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    return constant_cols

# Function to find duplicate columns
def find_duplicate_columns(df):
    """Find columns that are duplicates of others."""
    duplicate_cols = []
    # Compare all pairs of columns
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            # Only compare columns with the same data type
            if df[cols[i]].dtype == df[cols[j]].dtype:
                # Check if columns are identical
                if df[cols[i]].equals(df[cols[j]]):
                    duplicate_cols.append((cols[i], cols[j]))
    return duplicate_cols

