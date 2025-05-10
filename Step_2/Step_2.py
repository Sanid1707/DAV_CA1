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

#############################################################
# SECTION 7: CONSTANT/DUPLICATE DETECTION AND REMOVAL
#############################################################

# STEP 2.8: Check for constant columns and duplicates
constant_cols = find_constant_columns(df_cleaned)
print(f"\nConstant columns (to be dropped): {constant_cols}")

duplicate_cols = find_duplicate_columns(df_cleaned)
print(f"\nDuplicate columns: {duplicate_cols}")

# Create a table of dropped variables due to constants
constants_df = pd.DataFrame({
    'Variable': constant_cols,
    'Reason': ['Zero variance'] * len(constant_cols)
})

# Add duplicate columns
dup_info = []
for col1, col2 in duplicate_cols:
    dup_info.append({
        'Variable': col2,
        'Reason': f'Duplicate of {col1}'
    })
if dup_info:
    constants_df = pd.concat([constants_df, pd.DataFrame(dup_info)], ignore_index=True)

# Save to CSV
if not constants_df.empty:
    constants_df.to_csv('step2_dropped_constants.csv', index=False)
    print("Constants and duplicates saved to step2_dropped_constants.csv")

# Drop constant and duplicate columns
cols_to_drop = constant_cols + [col2 for col1, col2 in duplicate_cols]
df_cleaned = df_cleaned.drop(columns=cols_to_drop, errors='ignore')

#############################################################
# SECTION 8: OUTLIER ANALYSIS
#############################################################

# STEP 2.9: Outlier screening
# Select one key variable from each pillar for box plots
key_vars = []
for pillar, data in pillars.items():
    available = [var for var in data['vars'] if var in df_cleaned.columns]
    if available:
        key_vars.append(available[0])

# Create a panel of box plots
if key_vars:
    # Determine grid size
    n_vars = len(key_vars[:9])  # Max 9 plots
    n_cols = 3
    n_rows = (n_vars - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    axes = axes.flatten()
    
    for i, var in enumerate(key_vars[:9]):
        if i < len(axes):
            sns.boxplot(y=df_cleaned[var], ax=axes[i], color=sns.color_palette('viridis')[i % len(sns.color_palette('viridis'))])
            axes[i].set_title(f'Box Plot of {var}', fontsize=15, fontweight='bold')
            axes[i].set_ylabel('Value')
    
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Outlier Screening: Box Plots for Key Variables per Pillar', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make space for suptitle
    plt.savefig('images/step2_boxplot_panel.png', bbox_inches='tight')
    print("\nBox plot panel saved to images/step2_boxplot_panel.png")

#############################################################
# SECTION 9: COLLINEARITY ANALYSIS BY PILLAR
#############################################################

# STEP 2.10: Collinearity analysis WITHIN each pillar
# Initialize a dict to store vars to drop
vars_to_drop = set()

# Process each pillar separately
for pillar, data in pillars.items():
    pillar_vars = [var for var in data['vars'] if var in df_cleaned.columns]
    
    # Get numeric vars for this pillar
    num_pillar_vars = [var for var in pillar_vars if var in df_cleaned.select_dtypes(include=['number']).columns]
    
    if len(num_pillar_vars) < 2:  # Need at least 2 variables for correlation
        continue
        
    # Compute correlation matrix for this pillar
    pillar_corr = df_cleaned[num_pillar_vars].corr().abs()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(pillar_corr.columns)):
        for j in range(i+1, len(pillar_corr.columns)):
            if pillar_corr.iloc[i, j] > 0.9:
                var1 = pillar_corr.columns[i]
                var2 = pillar_corr.columns[j]
                corr = pillar_corr.iloc[i, j]
                high_corr_pairs.append((var1, var2, corr))
                
    print(f"\nHighly correlated pairs within {pillar}:")
    if high_corr_pairs:
        for var1, var2, corr in high_corr_pairs:
            print(f"{var1} and {var2}: {corr:.3f}")
            
            # IMPORTANT: NEVER drop Crime Indicator variables
            if pillar == 'Crime Indicators':
                # Skip this pair, don't drop crime indicators
                continue
                
            # For other pillars, drop the one with more missing values
            miss1 = missing_df_shortlist.loc[missing_df_shortlist['Variable'] == var1, 'Missing %'].values[0] if var1 in missing_df_shortlist['Variable'].values else 0
            miss2 = missing_df_shortlist.loc[missing_df_shortlist['Variable'] == var2, 'Missing %'].values[0] if var2 in missing_df_shortlist['Variable'].values else 0
            
            if miss1 > miss2:
                vars_to_drop.add(var1)
            else:
                vars_to_drop.add(var2)
    else:
        print("No variable pairs with correlation > 0.9")

print(f"\nVariables to drop due to high collinearity ({len(vars_to_drop)}):")
for var in vars_to_drop:
    print(f"  - {var}")

# STEP 2.11: Create final dataset after dropping variables
df_final = df_cleaned.drop(columns=list(vars_to_drop), errors='ignore')
print(f"\nFinal dataset shape after dropping collinear variables: {df_final.shape}")

# Final correlation heatmap of selected variables
plt.figure(figsize=(20, 18))  # Adjusted size

# Use a subset of numeric variables if there are too many
final_numeric_cols = df_final.select_dtypes(include=['number']).columns
if len(final_numeric_cols) > 50:  # Threshold for too many variables for annotated heatmap
    # Select representative variables from each pillar
    subset_vars = []
    for pillar_name, data in pillars.items():
        available = [var for var in data['vars'] if var in df_final.columns and var in final_numeric_cols]
        subset_vars.extend(available[:5])  # Take up to 5 from each pillar
    subset_vars = list(set(subset_vars))  # Ensure unique vars
    if len(subset_vars) < 2:  # ensure there are enough for a corr matrix
        subset_vars = final_numeric_cols[:20]  # fallback to first 20 numeric if subset is too small
    corr_final = df_final[subset_vars].corr()
    annotate_heatmap = False  # Turn off annotations if still too many
else:
    corr_final = df_final[final_numeric_cols].corr()
    annotate_heatmap = True

mask = np.triu(np.ones_like(corr_final, dtype=bool))
cmap_final = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)  # Consistent cmap
sns.heatmap(corr_final, mask=mask, cmap=cmap_final, vmax=1, vmin=-1, center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .6, "label": "Pearson Correlation"}, 
            annot=annotate_heatmap, fmt='.2f', annot_kws={"size": 8})
plt.title('Correlation Matrix of Final Selected Variables', fontsize=22, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout(pad=2.0)
plt.savefig('images/step2_corr_heatmap.png', bbox_inches='tight')
print("\nCorrelation heatmap of final variables saved to images/step2_corr_heatmap.png")

# STEP 2.12: Save final shortlist of variables
df_final.columns.to_series().to_csv('step2_selected_vars.csv', index=False, header=['Variable'])
print(f"\nFinal selection of {len(df_final.columns)} variables saved to step2_selected_vars.csv")

# Create summary statistics by pillar
data_characteristics = []

for pillar, data in pillars.items():
    pillar_vars = [var for var in data['vars'] if var in df_final.columns]
    if pillar_vars:
        # Calculate missing percentages for these variables before imputation
        miss_pcts = []
        hard_count = 0
        soft_count = 0
        proxy_count = 0
        
        for var in pillar_vars:
            if var in missing_df_shortlist['Variable'].values:
                miss_pct = missing_df_shortlist.loc[missing_df_shortlist['Variable'] == var, 'Missing %'].values[0]
                miss_pcts.append(miss_pct)
            else:
                miss_pcts.append(0)
            
            # Count types
            if var in df_final.select_dtypes(include=['number']).columns:
                hard_count += 1
            else:
                soft_count += 1
            
            # Count proxies
            if var in ['PctVacantBoarded', 'PctWOFullPlumb', 'PctUnemployed', 'medIncome']:
                proxy_count += 1
        
        # Add to summary table
        data_characteristics.append({
            'Pillar': pillar,
            'Variable Count': len(pillar_vars),
            'Avg Missing %': f"{np.mean(miss_pcts):.1f}%" if miss_pcts else "0.0%",
            'Hard Variables': hard_count,
            'Soft Variables': soft_count,
            'Proxy Variables': proxy_count
        })

# Convert to DataFrame
data_char_df = pd.DataFrame(data_characteristics)

#############################################################
# SECTION 10: SUMMARY VISUALIZATION AND STATISTICS
#############################################################

# Create a table visualization for the summary statistics
if not data_char_df.empty:
    fig, ax = plt.subplots(figsize=(14, len(data_char_df) * 0.6 + 1.5))  # Adjusted size
    ax.axis('tight')
    ax.axis('off')
    ax.set_title('Summary Statistics by Pillar (Final Selected Variables)', fontsize=20, fontweight='bold', pad=20)

    table = ax.table(
        cellText=data_char_df.values,
        colLabels=data_char_df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.12, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)  # Scale table

    # Style table cells - header and data rows
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor(sns.color_palette('viridis')[3])  # Darker header
        else:  # Data rows
            cell.set_facecolor('whitesmoke' if i % 2 == 0 else 'white')
        cell.set_edgecolor('grey')
        cell.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig('images/step2_pillar_summary.png', bbox_inches='tight')
    print("\nSummary statistics by pillar table image saved to images/step2_pillar_summary.png")
else:
    print("\nSummary statistics table is empty, not saving image.")

# Also save as CSV
if not data_char_df.empty:
    data_char_df.to_csv('step2_pillar_summary.csv', index=False)
    print("Pillar summary saved to step2_pillar_summary.csv")

# Calculate crime variable presence in final dataset
crime_indicators = [var for var in pillars['Crime Indicators']['vars'] if var in df_final.columns]
print(f"\nNumber of Crime Indicators in final dataset: {len(crime_indicators)}")
if crime_indicators:
    print("Crime Indicators retained:")
    for var in crime_indicators:
        print(f"  - {var}")

# Summary of findings
print("\n=== SUMMARY OF DATA SELECTION ANALYSIS ===")
print(f"Total variables after cleaning: {df_final.shape[1]}")
print(f"Variables dropped due to high missingness: {len(high_missing)}")
print(f"Variables dropped due to zero variance or duplicates: {len(cols_to_drop)}")
print(f"Variables dropped due to high collinearity: {len(vars_to_drop)}")
print(f"Average missing percentage: {missing_df_shortlist['Missing %'].mean():.2f}%")
print("\nPillar Representation in Final Dataset:")
for pillar, data in pillars.items():
    pillar_vars = [var for var in data['vars'] if var in df_final.columns]
    print(f"  - {pillar}: {len(pillar_vars)} variables")

print("\nRecommendations:")
print("1. Zeros in crime rate variables are now kept as valid observations")
print("2. Missingness thresholds are applied differently by pillar (40% for inputs, 60% for Crime Outputs)")
print("3. Collinearity pruning is now done within each pillar separately, preserving Crime Output variables")
print("4. All visualizations have been saved to the 'images' folder for reporting purposes")
print("5. Consider further refinement of crime variables to create a composite Crime-Risk score")

# STEP 2.8: Correlation network visualization
print("\nGenerating correlation network visualization...")
# Gather all highly correlated pairs across pillars
all_high_corr_pairs = []
for pillar, data in pillars.items():
    pillar_vars = [var for var in data['vars'] if var in df_cleaned.columns]
    num_pillar_vars = [var for var in pillar_vars if var in df_cleaned.select_dtypes(include=['number']).columns]
    
    if len(num_pillar_vars) < 2:
        continue
        
    pillar_corr = df_cleaned[num_pillar_vars].corr().abs()
    
    for i in range(len(pillar_corr.columns)):
        for j in range(i+1, len(pillar_corr.columns)):
            if pillar_corr.iloc[i, j] > 0.9:
                var1 = pillar_corr.columns[i]
                var2 = pillar_corr.columns[j]
                corr = pillar_corr.iloc[i, j]
                all_high_corr_pairs.append((var1, var2, corr, pillar))

# Create network visualization if we have correlated pairs
if all_high_corr_pairs:
    plt.figure(figsize=(20, 18))
    G = nx.Graph()
    
    # Collect all variables involved in high correlations
    all_corr_vars = set()
    for var1, var2, _, _ in all_high_corr_pairs:
        all_corr_vars.add(var1)
        all_corr_vars.add(var2)
    
    # Map variables to pillars
    var_to_pillar = {}
    for pillar_name, data in pillars.items():
        for var in data['vars']:
            if var in all_corr_vars:
                var_to_pillar[var] = pillar_name
    
    # Define distinct colors for different pillars
    distinct_colors = sns.color_palette("husl", n_colors=len(pillars) + 1)  # +1 for 'Other'
    pillar_colors_map = {name: distinct_colors[i] for i, name in enumerate(list(pillars.keys()) + ['Other'])}
    
    # Add nodes with pillar-based colors
    node_colors_list = []
    node_pillar_labels = {}
    for var in all_corr_vars:
        pillar_name = var_to_pillar.get(var, 'Other')
        node_colors_list.append(pillar_colors_map[pillar_name])
        node_pillar_labels[var] = pillar_name  # Store for legend
        G.add_node(var, pillar=pillar_name)
    
    # Add edges with correlation strength
    for var1, var2, corr, _ in all_high_corr_pairs:
        G.add_edge(var1, var2, weight=corr, title=f'{corr:.2f}')
    
    # Layout calculation
    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
    
    # Draw nodes with pillar-based colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=1500, alpha=0.85)
    
    # Draw edges with varying thickness based on correlation strength
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='grey')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Create legend for pillars
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=pillar_name, 
                             markersize=10, markerfacecolor=color) 
                     for pillar_name, color in pillar_colors_map.items() if pillar_name in set(node_pillar_labels.values())]
    
    plt.title('Network of Highly Correlated Variables (|r| > 0.9)', fontsize=22, fontweight='bold')
    plt.legend(handles=legend_handles, title='Variable Pillars', title_fontsize='14', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/step2_correlation_network.png', bbox_inches='tight')
    print("Correlation network visualization saved to images/step2_correlation_network.png")
else:
    print("Not enough highly correlated pairs to create a correlation network visualization.")

#############################################################
# SECTION 12: PROXY VALIDATION
#############################################################

# STEP 2.10: Proxy validation
print("\nPerforming proxy validation...")
# For socio-economic variables, use crime indicators as benchmarks
crime_benchmarks = ['ViolentCrimesPerPop', 'nonViolPerPop', 'murders', 'robberies']
available_benchmarks = [b for b in crime_benchmarks if b in df_final.columns]

if available_benchmarks:
    # Select a benchmark (prefer ViolentCrimesPerPop if available)
    benchmark = 'ViolentCrimesPerPop' if 'ViolentCrimesPerPop' in available_benchmarks else available_benchmarks[0]
    
    # Candidate proxy variables from socio-economic disadvantage
    proxy_candidates = ['PctUnemployed', 'PctPopUnderPov', 'PctLowIncomeUnderPov', 'medIncome', 
                        'PctKidsBornNevrMarr', 'PctIlleg', 'PctImmigRec5']
    available_proxies = [p for p in proxy_candidates if p in df_final.columns]
    
    if available_proxies and benchmark in df_final.select_dtypes(include=['number']).columns:
        # Calculate correlations between benchmark and proxy candidates
        proxy_cors = {}
        for proxy in available_proxies:
            if proxy in df_final.select_dtypes(include=['number']).columns:
                cor = df_final[[proxy, benchmark]].corr().iloc[0, 1]
                proxy_cors[proxy] = abs(cor)
        
        if proxy_cors:
            plt.figure(figsize=(14, 10))
            proxy_df = pd.DataFrame({
                'Proxy Variable': list(proxy_cors.keys()),
                'Correlation': list(proxy_cors.values())
            }).sort_values(by='Correlation', ascending=False).head(10)
            
            ax = sns.barplot(x='Correlation', y='Proxy Variable', data=proxy_df, palette='mako_r')
            plt.title(f'Proxy Validation: Correlation with {benchmark}', fontsize=18, fontweight='bold')
            plt.xlabel('Absolute Correlation Coefficient', fontsize=14)
            plt.ylabel('Socio-economic Proxy Variables', fontsize=14)
            
            # Add value labels
            for p in ax.patches:
                width = p.get_width()
                plt.text(width + 0.01, p.get_y() + p.get_height()/2 + 0.1,
                        f'{width:.3f}', ha='left', va='center', fontsize=12)
            
            plt.xlim(0, 1)  # Correlation is between 0 and 1 (absolute)
            plt.tight_layout()
            plt.savefig('images/step2_proxy_cors.png', bbox_inches='tight')
            print(f"Proxy validation visualization saved to images/step2_proxy_cors.png")
            
            # Save to CSV for reference
            proxy_df.to_csv('step2_proxy_correlations.csv', index=False)
        else:
            print("Could not calculate correlations between proxies and benchmark.")
    else:
        print("No suitable proxy variables or benchmark variable found for proxy validation.")
else:
    print("No crime benchmarks available for proxy validation.")

#############################################################
# SECTION 13: FINAL REPORT GENERATION AND DATASET EXPORT
#############################################################

# STEP 2.13: Generate Report Narrative
print("\nGenerating data selection report narrative...")

report_content = f"""# Data Selection Process (Step 2)

## Overview
This document details the data selection process for the crime risk analysis project, following a theory-driven approach based on social disorganization theory. The process ensured that all crime indicators were preserved while maintaining data quality and addressing redundancy issues.

## Step 2.0: Conceptual Mapping
Starting with all {df.shape[1]} variables in the dataset, each variable was mapped to one of four theoretical pillars:
- Socio-economic Disadvantage
- Residential Instability
- Population & Demographic factors
- Crime Indicators (outputs)

Each variable was assigned a role (Input, Process, or Output) based on its theoretical function. This mapping is available in `step2_variable_mapping.csv`.

## Step 2.1: Initial Shortlist
Based on the theoretical mapping, {len(shortlisted_vars)} variables were shortlisted for further analysis. The distribution across pillars was:
{pillar_counts.to_string(index=False)}

## Step 2.2: Missing Data Analysis
Missing data was analyzed for all shortlisted variables. The top 15 variables with missing data are visualized in `step2_missing_top15.png`. Instead of treating zeros as missing in crime variables, the analysis preserved legitimate zero values, which represent valid "no crime" observations.

## Step 2.3: Variable Type Classification
Variables were classified as either Hard (quantitative) or Soft (qualitative), and as Direct measures or Proxy measures. The distributions are visualized in `step2_var_type.png` and `step2_measure_type.png`.

## Step 2.4: Pillar Distribution
The distribution of variables across the theoretical pillars is shown in `step2_vars_by_pillar.png`, confirming good representation of all aspects of the theoretical framework.

## Step 2.5: Handling Missing Data
Different thresholds were applied for different variable types:
- 40% missing threshold for non-crime variables
- 60% missing threshold for crime variables

This approach ensured that important crime indicators were not unnecessarily excluded. Variables dropped due to high missingness are listed in `step2_dropped_missing.csv`.

## Step 2.6: Constant and Duplicate Detection
No constant columns or exact duplicates were found in the dataset. If any had been found, they would have been documented in `step2_dropped_constants.csv`.

## Step 2.7: Outlier Screening
Outliers were examined for key variables from each pillar, as shown in `step2_boxplot_panel.png`. This helped identify extreme values that might influence subsequent analysis.

## Step 2.8: Correlation Analysis Within Pillars
Correlation analysis was performed within each pillar separately, rather than across the entire dataset. This approach preserved the theoretical structure while still addressing multicollinearity. The network of highly correlated variables is visualized in `step2_correlation_network.png`.

## Step 2.9: Final Correlation Analysis
A correlation heatmap of the final selected variables is provided in `step2_corr_heatmap.png`, showing the relationships between variables after all cleaning steps.

## Step 2.10: Proxy Validation
Proxy variables for socio-economic disadvantage were validated against crime outcome measures. The correlation between proxy variables and the benchmark crime indicator is visualized in `step2_proxy_cors.png`.

## Step 2.11: Final Variable Selection
The final selection includes {df_final.shape[1]} variables that:
- Match the theoretical framework
- Have acceptable levels of missing data
- Are not constant or duplicate
- Are not excessively collinear within their pillar

The final variable list is available in `step2_selected_vars.csv`.

## Step 2.12: Summary by Pillar
The final dataset includes variables from all four theoretical pillars:
- {len([var for var in pillars['Socio-economic Disadvantage']['vars'] if var in df_final.columns])} Socio-economic Disadvantage variables
- {len([var for var in pillars['Residential Instability']['vars'] if var in df_final.columns])} Residential Instability variables
- {len([var for var in pillars['Population & Demographic']['vars'] if var in df_final.columns])} Population & Demographic variables
- {len([var for var in pillars['Crime Indicators']['vars'] if var in df_final.columns])} Crime Indicators

Detailed statistics by pillar are shown in `step2_pillar_summary.png` and available in `step2_pillar_summary.csv`.

## Next Steps
With the data now properly selected and cleaned, the next step is to:
1. Consider creating a composite Crime-Risk score from the crime indicators
2. Perform feature engineering and transformation as needed
3. Proceed with exploratory analysis of relationships between input variables and crime outcomes
"""

# Save the report
with open('step2_report.md', 'w') as f:
    f.write(report_content)

print("Data selection report saved to step2_report.md")

# Create a summary of the final selected variables by pillar with descriptions
print("\nCreating summary of final selected variables by pillar...")

# Create variable descriptions (if not already available)
var_descriptions = {
    # Socio-economic Disadvantage
    'PctUnemployed': 'Percentage of population unemployed',
    'PctPopUnderPov': 'Percentage of population under poverty level',
    'PctLowIncomeUnderPov': 'Percentage of low-income population under poverty level',
    'medIncome': 'Median household income',
    'PctFam2Par': 'Percentage of families with two parents',
    'PctKidsBornNevrMarr': 'Percentage of kids born to never married parents',
    'PctIlleg': 'Percentage of illegitimate children',
    'PctImmigRec5': 'Percentage of immigrants who immigrated within last 5 years',
    'PctImmigRec8': 'Percentage of immigrants who immigrated within last 8 years',
    'PctImmigRec10': 'Percentage of immigrants who immigrated within last 10 years',
    'PctRecentImmig': 'Percentage of population who are recent immigrants',
    'PctRecImmig5': 'Percentage of population who immigrated within last 5 years',
    'PctRecImmig8': 'Percentage of population who immigrated within last 8 years',
    'PctRecImmig10': 'Percentage of population who immigrated within last 10 years',
    'PctNotSpeakEnglWell': 'Percentage of population who do not speak English well',
    'PctLargHouseFam': 'Percentage of large family households',
    'PctLargHouseOccup': 'Percentage of large occupied households',
    'PersPerFam': 'Average number of persons per family',
    'PersPerOccupHous': 'Average number of persons per occupied household',
    'PersPerRentOccHous': 'Average number of persons per rented occupied household',
    
    # Residential Instability
    'PctVacantBoarded': 'Percentage of vacant housing that is boarded up',
    'PctHousNoPhone': 'Percentage of households without phone',
    'PctWOFullPlumb': 'Percentage of housing without full plumbing',
    'OwnOccQrange': 'Owner occupied housing - interquartile range',
    'RentQrange': 'Rent - interquartile range',
    'PctHousOccup': 'Percentage of housing occupied',
    'PctHousOwnerOccup': 'Percentage of housing owner occupied',
    'PctVacantHous': 'Percentage of housing vacant',
    'MedRentpctHousInc': 'Median rent as percentage of household income',
    'MedOwnCostpctInc': 'Median owner cost as percentage of household income',
    'PctSameHouse85': 'Percentage of population living in same house since 1985',
    'PctSameCounty85': 'Percentage of population living in same county since 1985',
    'PctMovedIn95': 'Percentage of population who moved in 1995',
    'NumInShelters': 'Number of people in homeless shelters',
    'NumStreet': 'Number of homeless people on streets',
    
    # Population & Demographic
    'population': 'Total population',
    'pctUrban': 'Percentage of population that is urban',
    'medAge': 'Median age',
    'pctWPubAsst': 'Percentage with public assistance',
    'pctWSocSec': 'Percentage with social security income',
    'racepctblack': 'Percentage of population that is African American',
    'racePctWhite': 'Percentage of population that is Caucasian',
    'racePctAsian': 'Percentage of population that is Asian',
    'racePctHisp': 'Percentage of population that is Hispanic',
    'pctForeignBorn': 'Percentage of population that is foreign born',
    'PctPersDenseHous': 'Percentage of persons in dense housing',
    'PctUsePubTrans': 'Percentage of people using public transportation',
    
    # Crime Indicators
    'murders': 'Number of murders',
    'rapes': 'Number of rapes',
    'robberies': 'Number of robberies',
    'assaults': 'Number of assaults',
    'burglaries': 'Number of burglaries',
    'larcenies': 'Number of larcenies',
    'autoTheft': 'Number of auto thefts',
    'arsons': 'Number of arsons',
    'murdPerPop': 'Number of murders per population',
    'rapesPerPop': 'Number of rapes per population',
    'robbbPerPop': 'Number of robberies per population',
    'assaultPerPop': 'Number of assaults per population',
    'burglPerPop': 'Number of burglaries per population',
    'larcPerPop': 'Number of larcenies per population',
    'autoTheftPerPop': 'Number of auto thefts per population',
    'arsonsPerPop': 'Number of arsons per population',
    'ViolentCrimesPerPop': 'Number of violent crimes per population',
    'nonViolPerPop': 'Number of non-violent crimes per population',
    
    # Identifiers
    'communityname': 'Name of the community',
    'state': 'State abbreviation',
    'countyCode': 'County code',
    'communityCode': 'Community code'
}

# Create a default description for any variable not in the dictionary
for var in df_final.columns:
    if var not in var_descriptions:
        var_descriptions[var] = f"Measures {var.lower()} in the community"

# Create a dictionary to hold variables by pillar
pillar_vars = {}
for pillar, data in pillars.items():
    pillar_vars[pillar] = [var for var in data['vars'] if var in df_final.columns]

# Add identifiers as a separate category
pillar_vars["Identifiers"] = [var for var in df_final.columns if var in ['communityname', 'state', 'countyCode', 'communityCode']]

# Remaining variables that don't fit in any pillar
all_categorized = []
for vars_list in pillar_vars.values():
    all_categorized.extend(vars_list)
pillar_vars["Other"] = [var for var in df_final.columns if var not in all_categorized]

# Create a dataframe of variables by pillar with descriptions
pillar_var_desc = []
for pillar, vars_list in pillar_vars.items():
    if vars_list:  # Only include pillars with variables
        for var in vars_list:
            pillar_var_desc.append({
                'Pillar': pillar,
                'Variable': var,
                'Description': var_descriptions.get(var, f"Measures {var.lower()} in the community")
            })

pillar_var_df = pd.DataFrame(pillar_var_desc)

# Save to CSV
pillar_var_df.to_csv('step2_variables_by_pillar.csv', index=False)
print("Variables by pillar with descriptions saved to step2_variables_by_pillar.csv")

# Also create a markdown table for the report
md_table = "## Selected Variables by Pillar\n\n"
for pillar in pillar_vars.keys():
    vars_in_pillar = [var for var in pillar_var_df['Variable'] if pillar_var_df[pillar_var_df['Variable'] == var]['Pillar'].values[0] == pillar]
    
    if vars_in_pillar:
        md_table += f"\n### {pillar}\n\n"
        md_table += "| Variable | Description |\n"
        md_table += "|----------|-------------|\n"
        
        for var in vars_in_pillar:
            desc = pillar_var_df[pillar_var_df['Variable'] == var]['Description'].values[0]
            md_table += f"| {var} | {desc} |\n"

# Save the markdown table
with open('step2_variables_by_pillar.md', 'w') as f:
    f.write(md_table)
print("Markdown table of variables by pillar saved to step2_variables_by_pillar.md")


#############################################################
# SECTION 14: FINAL DATASET EXPORT
#############################################################


# Ensure community name is included in the final dataset (keep as a backup)
if 'communityname' not in df_final.columns and 'communityname' in df.columns:
    # Get community name from original dataset
    df_final = df_final.copy()
    df_final['communityname'] = df['communityname']
    print("Added community name to final dataset")
elif 'communityname' in df_final.columns:
    print("Community name already included in final dataset")
else:
    print("Warning: Community name column not found in original dataset")

# Save the complete cleaned dataset for further processing
df_final.to_csv('step2_final_dataset.csv', index=False)
print("Complete cleaned dataset saved to step2_final_dataset.csv")

#############################################################
# END OF CRIME DATA ANALYSIS SCRIPT
#############################################################
