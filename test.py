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

#############################################################
# SECTION 3: INITIAL DATA PROCESSING
#############################################################

print("Cleaning and preparing the dataset...")
# Apply data cleaning steps
df = fix_column_names(df)
df = handle_question_marks(df)
df = convert_to_numeric(df)
df = handle_implausible_zeros(df)

# Display basic information about the dataset
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows of the dataset:")
print(df.head())

# Calculate missing percentages
missing_df = missing_percentage(df)
print("\nMissing percentage for each variable:")
print(missing_df)

#############################################################
# SECTION 4: THEORETICAL FRAMEWORK AND VARIABLE MAPPING
#############################################################

# STEP 2.0: Theory-based mapping of all variables
# Define pillars and map variables to them with roles
pillars = {
    'Socio-economic Disadvantage': {
        'vars': ['PctUnemployed', 'PctPopUnderPov', 'PctLowIncomeUnderPov', 'medIncome', 'PctFam2Par',
                'PctKidsBornNevrMarr', 'PctIlleg', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
                'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctNotSpeakEnglWell',
                'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerFam', 'PersPerOccupHous', 'PersPerRentOccHous'],
        'role': 'Input'
    },
    'Residential Instability': {
        'vars': ['PctVacantBoarded', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccQrange', 'RentQrange', 
                'PctHousOccup', 'PctHousOwnerOccup', 'PctVacantHous', 'MedRentpctHousInc', 'MedOwnCostpctInc', 
                'PctSameHouse85', 'PctSameCounty85', 'PctMovedIn95', 'NumInShelters', 'NumStreet'],
        'role': 'Input'
    },
    'Population & Demographic': {
        'vars': ['population', 'pctUrban', 'medAge', 'pctWPubAsst', 'pctWSocSec', 'racepctblack', 'racePctWhite', 
                'racePctAsian', 'racePctHisp', 'pctForeignBorn', 'PctPersDenseHous', 'PctUsePubTrans'],
        'role': 'Process'
    },
    'Crime Indicators': {
        'vars': ['murders', 'rapes', 'robberies', 'assaults', 'burglaries', 'larcenies', 'autoTheft', 'arsons',
                'murdPerPop', 'rapesPerPop', 'robbbPerPop', 'assaultPerPop', 'burglPerPop', 'larcPerPop', 
                'autoTheftPerPop', 'arsonsPerPop', 'ViolentCrimesPerPop', 'nonViolPerPop'],
        'role': 'Output'
    }
}

# Create a variable mapping dataframe
variable_map = pd.DataFrame({
    'Variable': df.columns,
    'Description': [f"Measures {col.lower()} in the community" for col in df.columns],
    'Keep?': 'N'  # Default is to not keep
})

# Explicitly mark community name to be kept
if 'communityname' in variable_map['Variable'].values:
    community_idx = variable_map.index[variable_map['Variable'] == 'communityname'].tolist()[0]
    variable_map.loc[community_idx, 'Keep?'] = 'Y'
    variable_map.loc[community_idx, 'Pillar'] = 'Identifier'
    variable_map.loc[community_idx, 'Role'] = 'Identifier'
    print("Community name marked to be kept throughout analysis")

# Assign pillars and roles to variables
variable_map['Pillar'] = 'Other'
variable_map['Role'] = 'Other'

# Mark variables to keep based on pillars
for pillar, data in pillars.items():
    for var in data['vars']:
        idx = variable_map.index[variable_map['Variable'] == var].tolist()
        if idx:
            variable_map.loc[idx[0], 'Pillar'] = pillar
            variable_map.loc[idx[0], 'Role'] = data['role']
            variable_map.loc[idx[0], 'Keep?'] = 'Y'  # Mark to keep

# Save the variable mapping
variable_map.to_csv('step2_variable_mapping.csv', index=False)
print("\nVariable mapping created and saved to step2_variable_mapping.csv")

# STEP 2.1: Initial theory-driven shortlist
# Keep only those with "Keep?=Y"
shortlisted_vars = variable_map[variable_map['Keep?'] == 'Y']['Variable'].tolist()
df_shortlist = df[shortlisted_vars]

# Count variables by pillar
pillar_counts = variable_map[variable_map['Keep?'] == 'Y'].groupby('Pillar').size().reset_index(name='Count')
pillar_counts.to_csv('step2_shortlist_counts.csv', index=False)
print(f"\nInitial theory-driven shortlist created with {len(shortlisted_vars)} variables")
print(pillar_counts)

#############################################################
# SECTION 5: VARIABLE ANALYSIS AND CLASSIFICATION
#############################################################

# STEP 2.2: Calculate missing percentages after cleanup
missing_df_shortlist = missing_percentage(df_shortlist)

# Plot top 15 missing variables
plt.figure(figsize=(14, 10))
missing_plot_data = missing_df_shortlist.head(15).sort_values('Missing %')
ax = sns.barplot(x='Missing %', y='Variable', data=missing_plot_data, palette='mako')
plt.title('Top 15 Variables with Highest Missing Data', fontsize=18, fontweight='bold')
plt.xlabel('Missing Percentage (%)')
plt.ylabel('Variable')
# Add value labels to bars
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 0.5, p.get_y() + p.get_height()/2. + 0.1,
             f'{width:.1f}%', ha='left', va='center')
plt.tight_layout()
plt.savefig('images/step2_missing_top15.png', bbox_inches='tight')
print("\nMissing data visualization saved to images/step2_missing_top15.png")

# STEP 2.3: Type & proxy summary
# Define variable types and proxy status
var_types = {}
proxy_status = {}

for var in df_shortlist.columns:
    # Determine variable type (Hard/Soft)
    if var in df_shortlist.select_dtypes(include=['number']).columns:
        var_types[var] = 'Hard (Quantitative)'
    else:
        var_types[var] = 'Soft (Qualitative)'
    
    # Determine proxy status
    if var in ['PctVacantBoarded', 'PctWOFullPlumb', 'PctUnemployed', 'medIncome']:
        proxy_status[var] = 'Proxy'
    else:
        proxy_status[var] = 'Direct'

# Add to variable mapping
variable_map['Type'] = variable_map['Variable'].map(var_types).fillna('Unknown')
variable_map['Measurement'] = variable_map['Variable'].map(proxy_status).fillna('Direct')

# Plot variable types
plt.figure(figsize=(10, 7))
type_counts = variable_map[variable_map['Keep?'] == 'Y']['Type'].value_counts()
ax = sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis') # Swapped x and y for vertical bars
plt.title('Distribution of Variable Types (Hard vs. Soft)', fontsize=18, fontweight='bold')
plt.xlabel('Variable Type')
plt.ylabel('Number of Variables')
# Add value labels to bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.5,
            f'{height:.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/step2_var_type.png', bbox_inches='tight')
print("\nVariable type distribution saved to images/step2_var_type.png")

# Plot proxy vs direct measures
plt.figure(figsize=(10, 7))
proxy_counts = variable_map[variable_map['Keep?'] == 'Y']['Measurement'].value_counts()
ax = sns.barplot(x=proxy_counts.index, y=proxy_counts.values, palette='viridis') # Swapped x and y
plt.title('Distribution of Direct vs. Proxy Measures', fontsize=18, fontweight='bold')
plt.xlabel('Measurement Type')
plt.ylabel('Number of Variables')
# Add value labels to bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.5,
            f'{height:.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/step2_measure_type.png', bbox_inches='tight')
print("\nProxy vs direct measures visualization saved to images/step2_measure_type.png")

# STEP 2.4: Pillar breakdown
plt.figure(figsize=(14, 10))
pillar_var_counts = variable_map[variable_map['Keep?'] == 'Y'].groupby('Pillar').size().sort_values(ascending=False)
ax = sns.barplot(x=pillar_var_counts.index, y=pillar_var_counts.values, palette='crest_r') # Swapped x and y
plt.title('Number of Variables per Pillar (Post Initial Shortlist)', fontsize=18, fontweight='bold')
plt.xlabel('Pillar')
plt.ylabel('Number of Variables')
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
# Add value labels to bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 0.5,
            f'{height:.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/step2_vars_by_pillar.png', bbox_inches='tight')
print("\nVariables by pillar visualization saved to images/step2_vars_by_pillar.png")

#############################################################
# SECTION 6: MISSING DATA HANDLING AND THRESHOLDS
#############################################################

# STEP 2.5: Apply different missing thresholds by pillar
# Higher threshold for Crime Indicators (60%) vs. other pillars (40%)
high_missing = []
crime_indicators = [var for var in pillars['Crime Indicators']['vars'] if var in df_shortlist.columns]

# Split into crime and non-crime variables
crime_vars_missing = missing_df_shortlist[missing_df_shortlist['Variable'].isin(crime_indicators)]
non_crime_vars_missing = missing_df_shortlist[~missing_df_shortlist['Variable'].isin(crime_indicators)]

# Apply appropriate thresholds
crime_high_missing = crime_vars_missing[crime_vars_missing['Missing %'] > 60]['Variable'].tolist()
non_crime_high_missing = non_crime_vars_missing[non_crime_vars_missing['Missing %'] > 40]['Variable'].tolist()

# Combine the lists
high_missing = crime_high_missing + non_crime_high_missing

print(f"\nVariables with more than 40% missing (non-crime) or 60% missing (crime) to be dropped: {high_missing}")

# Save dropped variables due to high missingness
high_missing_df = pd.concat([
    crime_vars_missing[crime_vars_missing['Missing %'] > 60],
    non_crime_vars_missing[non_crime_vars_missing['Missing %'] > 40]
])
high_missing_df.to_csv('step2_dropped_missing.csv', index=False)
print("High missing variables saved to step2_dropped_missing.csv")

# Drop variables with high missing percentages
df_cleaned = df_shortlist.drop(columns=high_missing)

# STEP 2.6: Impute remaining missing values by pillar
# For this example, we'll use median imputation within each pillar
for pillar, data in pillars.items():
    pillar_vars = [var for var in data['vars'] if var in df_cleaned.columns]
    num_pillar_vars = [var for var in pillar_vars if var in df_cleaned.select_dtypes(include=['number']).columns]
    
    if num_pillar_vars:
        print(f"Imputing missing values for {pillar} variables using median imputation...")
        # Impute numeric variables with median
        df_cleaned[num_pillar_vars] = df_cleaned[num_pillar_vars].fillna(df_cleaned[num_pillar_vars].median())

# Create a complete correlation matrix of all variables after pruning for missing data
numeric_cols_after_missing = df_cleaned.select_dtypes(include=['number']).columns
corr_after_missing = df_cleaned[numeric_cols_after_missing].corr()

# STEP 2.7: Generate a heatmap visualization
plt.figure(figsize=(28, 24))  # Adjusted figure size
mask = np.triu(np.ones_like(corr_after_missing, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)  # Professional diverging palette

sns.heatmap(corr_after_missing, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .6, "label": "Pearson Correlation"})

plt.title('Complete Correlation Matrix (Post Missing Data Pruning, Pre-Collinearity Removal)', fontsize=22, fontweight='bold')
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8)
plt.tight_layout(pad=2.0)
plt.savefig('images/step2_complete_correlation_after_missing.png', bbox_inches='tight')
print("\nComplete correlation matrix after missing data pruning saved to images/step2_complete_correlation_after_missing.png")

# Create a clustered correlation matrix for better pattern visualization
clustergrid = sns.clustermap(
    corr_after_missing, 
    cmap=cmap,  # Use the same professional cmap
    center=0,
    linewidths=.5, 
    figsize=(28, 28),  # Adjusted figure size
    dendrogram_ratio=(0.05, 0.05),  # Adjust dendrogram ratios
    cbar_pos=(0.02, 0.8, 0.03, 0.18),  # Reposition colorbar
    cbar_kws={"label": "Pearson Correlation"},
    xticklabels=True, yticklabels=True
)
clustergrid.ax_heatmap.set_title('Clustered Correlation Matrix (Post Missing Data Pruning)', fontsize=22, fontweight='bold', pad=20)
clustergrid.ax_heatmap.tick_params(axis='x', labelsize=8, rotation=90)
clustergrid.ax_heatmap.tick_params(axis='y', labelsize=8)
plt.savefig('images/step2_clustered_correlation_after_missing.png', bbox_inches='tight')
print("\nClustered correlation matrix after missing data pruning saved to images/step2_clustered_correlation_after_missing.png")
