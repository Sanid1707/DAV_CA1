#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Multivariate Analysis
- Goal: Analyze relationships between indicators within pillars
- Identify key components through PCA
- Examine country clusters 
- Assess indicator collinearity
- Make decisions on indicator retention/removal
"""

# Monkey patch threadpool_limits to avoid issues with AttributeError
# This happens before importing any sklearn libraries
import sys
from unittest.mock import MagicMock
sys.modules['threadpoolctl'] = MagicMock()
import sklearn.utils.fixes
sklearn.utils.fixes.threadpool_limits = MagicMock()

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Ignore warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('Step_4/data', exist_ok=True)
os.makedirs('Step_4/output', exist_ok=True)
os.makedirs('Step_4/output/figures', exist_ok=True)

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

# Define a consistent color palette for pillars
pillar_colors = {
    'Demographics': '#440154',
    'Income': '#3b528b',
    'Housing': '#21918c',
    'Social': '#5ec962',
    'Crime': '#fde725'
}

print("\n" + "="*80)
print("STEP 4: MULTIVARIATE ANALYSIS")
print("="*80)

#############################################################
# SECTION 2: DATA LOADING AND PREPROCESSING
# - Load data from Step 3
# - Organize indicators into pillars
# - Standardize indicators
#############################################################

print("\n" + "="*80)
print("SECTION 2: DATA LOADING AND PREPROCESSING")
print("="*80)

# Load the final dataset from Step 3
print("\nLoading dataset from Step 3...")
try:
    df = pd.read_csv('step3_final_dataset_for_multivariate.csv')
    print(f"Successfully loaded dataset with {df.shape[0]} communities and {df.shape[1]} variables")
except FileNotFoundError:
    try:
        # Alternative file name
        df = pd.read_csv('step3_imputed_dataset.csv')
        print(f"Successfully loaded dataset with {df.shape[0]} communities and {df.shape[1]} variables")
    except FileNotFoundError:
        try:
            # Fallback to original file
            df = pd.read_csv('step3_outlier_handled_dataset.csv')
            print(f"Successfully loaded dataset with {df.shape[0]} communities and {df.shape[1]} variables")
        except FileNotFoundError:
            # Final fallback
            df = pd.read_csv('step3_final_dataset_for_multivariate.csv')
            print(f"Successfully loaded dataset with {df.shape[0]} communities and {df.shape[1]} variables")

# Display basic information about the dataset
print("\nDataset information:")
print(f"Number of communities: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")
print(f"Number of missing values: {df.isna().sum().sum()}")

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

# Save a copy of the original data
df_original = df.copy()

# 4-A: Z-standardize all numeric indicators within each pillar
print("\n4-A: Z-standardizing indicators within each pillar...")

# Create a standardized dataframe
standardized_df = df.copy()

# Initialize a dictionary to store standardized data for each pillar
pillar_data = {}

# Function to check if a variable is heavily skewed
def is_skewed(series, threshold=1.0):
    """Check if a series is heavily skewed based on skewness value"""
    skewness = series.skew()
    return abs(skewness) > threshold

# Function to standardize a variable
def standardize_variable(series, log_transform=False):
    """Standardize a variable, optionally with log transformation for skewed data"""
    if log_transform and (series > 0).all():
        # Apply log transformation for skewed positive data
        transformed = np.log(series)
        # Then standardize
        return (transformed - transformed.mean()) / transformed.std()
    else:
        # Standard z-score normalization
        return (series - series.mean()) / series.std()

# Standardize each pillar
for pillar, variables in pillars.items():
    print(f"\nProcessing {pillar} pillar...")
    
    # Filter valid variables (those that exist in the dataset)
    valid_vars = [var for var in variables if var in df.columns]
    print(f"  Valid variables: {len(valid_vars)}/{len(variables)}")
    
    # Create a dataframe with just these variables
    pillar_df = df[valid_vars].copy()
    
    # Check for and handle skewed variables
    for var in valid_vars:
        if is_skewed(pillar_df[var]):
            print(f"  Variable {var} is skewed, applying log-z standardization")
            standardized_df[var] = standardize_variable(pillar_df[var], log_transform=True)
        else:
            print(f"  Variable {var} is not heavily skewed, applying standard z-score")
            standardized_df[var] = standardize_variable(pillar_df[var], log_transform=False)
    
    # Store the standardized data for this pillar
    pillar_data[pillar] = standardized_df[valid_vars].copy()

# Save the standardized dataframe
standardized_df.to_csv('Step_4/data/step4_standardized_data.csv', index=False)
print("\nStandardized data saved to 'Step_4/data/step4_standardized_data.csv'")

# Create a function to create a simple standardization summary
def create_standardization_summary():
    summary_rows = []
    
    for pillar, variables in pillars.items():
        valid_vars = [var for var in variables if var in df.columns]
        
        for var in valid_vars:
            original_mean = df[var].mean()
            original_std = df[var].std()
            standardized_mean = standardized_df[var].mean()
            standardized_std = standardized_df[var].std()
            
            summary_rows.append({
                'Pillar': pillar,
                'Variable': var,
                'Original_Mean': original_mean,
                'Original_Std': original_std,
                'Standardized_Mean': standardized_mean,
                'Standardized_Std': standardized_std
            })
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

# Create and save standardization summary
standardization_summary = create_standardization_summary()
standardization_summary.to_csv('Step_4/data/step4_standardization_summary.csv', index=False)
print("Standardization summary saved to 'Step_4/data/step4_standardization_summary.csv'")

#############################################################
# SECTION 3: PRINCIPAL COMPONENT ANALYSIS (PCA)
# - Perform PCA on each pillar
# - Generate scree plots and loading tables
# - Determine optimal number of components
#############################################################

print("\n" + "="*80)
print("SECTION 3: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# 4-B: Exploratory PCA per pillar
print("\n4-B: Performing exploratory PCA for each pillar...")

# Initialize dictionaries to store PCA results
pca_models = {}
pca_results = {}
pca_loadings = {}
pca_variance = {}
pca_components_to_keep = {}

# Create summary dataframes for eigenvalues and variance explained
eigenvalue_summary = []
loadings_summary = []

# Function to determine number of components to keep based on Kaiser criterion and variance explained
def determine_components_to_keep(explained_variance_ratio):
    """
    Determine how many components to keep based on:
    1. Kaiser criterion (eigenvalue > 1)
    2. Cumulative variance explained >= 60%
    """
    # Calculate eigenvalues (for n=1 samples)
    eigenvalues = explained_variance_ratio * len(explained_variance_ratio)
    
    # Kaiser criterion: eigenvalue > 1
    kaiser_components = sum(eigenvalues > 1)
    
    # Cumulative variance >= 60%
    cum_variance = np.cumsum(explained_variance_ratio)
    variance_components = np.argmax(cum_variance >= 0.6) + 1  # +1 because of zero indexing
    
    # If no component explains 60%, take all components
    if variance_components == 0:
        variance_components = len(explained_variance_ratio)
    
    # Take the maximum of the two criteria
    return max(kaiser_components, variance_components)

# Perform PCA for each pillar
for pillar, variables in pillars.items():
    print(f"\nPerforming PCA for {pillar} pillar...")
    
    # Filter valid variables
    valid_vars = [var for var in variables if var in standardized_df.columns]
    
    if len(valid_vars) < 2:
        print(f"  Warning: Not enough variables in {pillar} pillar for PCA. Skipping.")
        continue
    
    # Extract the standardized data for this pillar
    X = standardized_df[valid_vars].values
    
    # Fit PCA
    pca = PCA()
    pca.fit(X)
    
    # Extract results
    explained_variance_ratio = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    loadings = pca.components_
    
    # Store results
    pca_models[pillar] = pca
    pca_results[pillar] = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': np.cumsum(explained_variance_ratio),
        'eigenvalues': eigenvalues
    }
    pca_loadings[pillar] = pd.DataFrame(
        loadings.T,
        index=valid_vars,
        columns=[f'PC{i+1}' for i in range(len(valid_vars))]
    )
    
    # Store variance explained
    pca_variance[pillar] = explained_variance_ratio
    
    # Determine number of components to keep
    components_to_keep = determine_components_to_keep(explained_variance_ratio)
    pca_components_to_keep[pillar] = components_to_keep
    
    print(f"  Number of components in {pillar} pillar: {len(valid_vars)}")
    print(f"  Variance explained by first component: {explained_variance_ratio[0]:.4f}")
    print(f"  Components to keep based on criteria: {components_to_keep}")
    
    # Add to the eigenvalue summary
    for i, (eigenvalue, variance, cum_variance) in enumerate(zip(
            eigenvalues, 
            explained_variance_ratio, 
            np.cumsum(explained_variance_ratio)
        )):
        eigenvalue_summary.append({
            'Pillar': pillar,
            'Component': f'PC{i+1}',
            'Eigenvalue': eigenvalue,
            'Variance_Explained': variance,
            'Cumulative_Variance': cum_variance,
            'Keep': i < components_to_keep
        })
    
    # Add to the loadings summary
    for var in valid_vars:
        for i in range(min(5, len(valid_vars))):  # Limit to first 5 components
            loadings_summary.append({
                'Pillar': pillar,
                'Variable': var,
                'Component': f'PC{i+1}',
                'Loading': pca_loadings[pillar].loc[var, f'PC{i+1}']
            })
    
    # Generate scree plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, color=pillar_colors.get(pillar, '#440154'))
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)  # Kaiser criterion line
    plt.title(f'Scree Plot - {pillar} Pillar')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(1, len(eigenvalues) + 1))
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
             np.cumsum(explained_variance_ratio), 
             'o-', color=pillar_colors.get(pillar, '#440154'))
    plt.axhline(y=0.6, color='r', linestyle='--', alpha=0.7)  # 60% variance threshold
    plt.title(f'Cumulative Variance - {pillar} Pillar')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/pca_scree_{pillar}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate loadings heatmap for components to keep
    if components_to_keep > 0:
        plt.figure(figsize=(12, 10))
        
        # Extract loadings for the components to keep
        loadings_to_plot = pca_loadings[pillar].iloc[:, :components_to_keep]
        
        # Sort variables by absolute loading on the first component
        sorted_vars = loadings_to_plot.iloc[:, 0].abs().sort_values(ascending=False).index
        loadings_to_plot = loadings_to_plot.loc[sorted_vars]
        
        # Generate heatmap
        sns.heatmap(loadings_to_plot, cmap='coolwarm', center=0, annot=True, fmt='.2f', 
                    cbar_kws={'label': 'Loading'})
        plt.title(f'Component Loadings - {pillar} Pillar')
        plt.tight_layout()
        plt.savefig(f'Step_4/output/figures/pca_loadings_{pillar}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate component interpretation visualization
        plt.figure(figsize=(14, components_to_keep * 4))
        
        for i in range(components_to_keep):
            plt.subplot(components_to_keep, 1, i+1)
            
            # Extract loadings for this component
            component_loadings = pca_loadings[pillar][f'PC{i+1}']
            
            # Sort by absolute loading
            sorted_loadings = component_loadings.abs().sort_values(ascending=False)
            sorted_vars = sorted_loadings.index
            
            # Plot horizontal bar chart
            colors = ['red' if component_loadings[var] < 0 else 'blue' for var in sorted_vars]
            plt.barh(sorted_vars, component_loadings[sorted_vars], color=colors)
            
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.title(f'PC{i+1} Loadings - {pillar} Pillar')
            plt.xlabel('Loading')
            
            # Add a legend
            plt.legend([Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='red', lw=4)],
                       ['Positive Loading', 'Negative Loading'])
            
        plt.tight_layout()
        plt.savefig(f'Step_4/output/figures/pca_interpretation_{pillar}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Create and save PCA eigenvalue and loadings summary tables
eigenvalue_df = pd.DataFrame(eigenvalue_summary)
loadings_df = pd.DataFrame(loadings_summary)

eigenvalue_df.to_csv('Step_4/data/pca_eigenvalues_summary.csv', index=False)
loadings_df.to_csv('Step_4/data/pca_loadings_summary.csv', index=False)

print("\nPCA eigenvalues summary saved to 'Step_4/data/pca_eigenvalues_summary.csv'")
print("PCA loadings summary saved to 'Step_4/data/pca_loadings_summary.csv'")

# Create a consolidated table of eigenvalues for all pillars
eigenvalue_pivot = eigenvalue_df.pivot_table(
    index=['Pillar', 'Component'],
    values=['Eigenvalue', 'Variance_Explained', 'Cumulative_Variance', 'Keep'],
    aggfunc='first'
).reset_index()

print("\nEigenvalue summary table:")
print(eigenvalue_pivot.head())

#############################################################
# SECTION 4: COMPONENT SELECTION AND INTERPRETATION
# - Decide on number of components to keep
# - Interpret the meaning of retained components
# - Generate component scores for further analysis
#############################################################

print("\n" + "="*80)
print("SECTION 4: COMPONENT SELECTION AND INTERPRETATION")
print("="*80)

# 4-C: Decide how many PCs to keep for each pillar
print("\n4-C: Deciding how many principal components to keep...")

# Initialize dictionary to store components to keep with rationale
components_rationale = {}

# Create a dictionary to store transformed data (component scores)
transformed_data = {}

# Decide and document for each pillar
for pillar in pca_models.keys():
    print(f"\nEvaluating {pillar} pillar...")
    
    # Extract PCA results
    eigenvalues = pca_results[pillar]['eigenvalues']
    explained_variance = pca_results[pillar]['explained_variance_ratio']
    cumulative_variance = pca_results[pillar]['cumulative_variance']
    components_to_keep = pca_components_to_keep[pillar]
    
    # Document the rationale
    kaiser_criterion = sum(eigenvalues > 1)
    cumulative_60_criterion = np.argmax(cumulative_variance >= 0.6) + 1
    
    rationale = f"""
    Number of components to keep for {pillar} pillar: {components_to_keep}
    
    Rationale:
    - Kaiser criterion (eigenvalue > 1): {kaiser_criterion} component(s)
    - 60% cumulative variance criterion: {cumulative_60_criterion} component(s)
    - Scree plot elbow: visual inspection shows an elbow at component {min(kaiser_criterion+1, len(eigenvalues))}
    - Conceptual interpretability: The retained components can be interpreted as follows:
    """
    
    # Add interpretations for each retained component
    for i in range(components_to_keep):
        # Get the top contributing variables (by absolute loading)
        loadings = pca_loadings[pillar][f'PC{i+1}']
        sorted_loadings = loadings.abs().sort_values(ascending=False)
        top_vars = sorted_loadings.index[:3]  # Top 3 variables
        
        # Add interpretation based on top variables and their loadings
        component_interp = f"\n    PC{i+1} ({explained_variance[i]*100:.1f}% variance): "
        
        # Add positive contributors
        pos_vars = [var for var in top_vars if loadings[var] > 0]
        if pos_vars:
            component_interp += f"Positively influenced by {', '.join(pos_vars)}; "
        
        # Add negative contributors
        neg_vars = [var for var in top_vars if loadings[var] < 0]
        if neg_vars:
            component_interp += f"Negatively influenced by {', '.join(neg_vars)}"
        
        rationale += component_interp
    
    # Store the rationale
    components_rationale[pillar] = rationale
    print(rationale)
    
    # Transform data to get component scores
    valid_vars = [var for var in pillars[pillar] if var in standardized_df.columns]
    X = standardized_df[valid_vars].values
    
    # Generate principal component scores
    X_transformed = pca_models[pillar].transform(X)
    
    # Keep only the selected components
    X_transformed = X_transformed[:, :components_to_keep]
    
    # Create a DataFrame with the component scores
    pc_columns = [f'{pillar}_PC{i+1}' for i in range(components_to_keep)]
    transformed_df = pd.DataFrame(X_transformed, columns=pc_columns)
    
    # Add community names if available
    if 'communityname' in df.columns:
        transformed_df['communityname'] = df['communityname']
    
    # Store transformed data
    transformed_data[pillar] = transformed_df

# Save component selection rationale
with open('Step_4/output/component_selection_rationale.txt', 'w') as f:
    for pillar, rationale in components_rationale.items():
        f.write(f"\n{rationale}\n{'='*80}\n")

print("\nComponent selection rationale saved to 'Step_4/output/component_selection_rationale.txt'")

# Create a combined DataFrame with all component scores
all_components_df = pd.DataFrame()

# Add community identifiers if available
if 'communityname' in df.columns:
    all_components_df['communityname'] = df['communityname']

# Add component scores from each pillar
for pillar, transformed_df in transformed_data.items():
    # Drop communityname if it exists to avoid duplication
    if 'communityname' in transformed_df.columns:
        component_columns = [col for col in transformed_df.columns if col != 'communityname']
        all_components_df = pd.concat([all_components_df, transformed_df[component_columns]], axis=1)
    else:
        all_components_df = pd.concat([all_components_df, transformed_df], axis=1)

# Save the combined component scores
all_components_df.to_csv('Step_4/data/all_component_scores.csv', index=False)
print("\nCombined component scores saved to 'Step_4/data/all_component_scores.csv'")

#############################################################
# SECTION 5: CLUSTER ANALYSIS ON COMMUNITIES
# - Perform hierarchical and k-means clustering
# - Determine optimal number of clusters
# - Visualize and interpret clusters
#############################################################

print("\n" + "="*80)
print("SECTION 5: CLUSTER ANALYSIS ON COMMUNITIES")
print("="*80)

# 4-D: Cluster analysis on communities using the retained PC scores
print("\n4-D: Performing cluster analysis on communities...")

# Initialize dictionaries to store clustering results
hierarchical_clusters = {}
kmeans_clusters = {}
cluster_results = {}

# Go through each pillar and perform clustering
for pillar, transformed_df in transformed_data.items():
    print(f"\nClustering communities based on {pillar} components...")
    
    # Get just the component scores (exclude identifier columns)
    component_columns = [col for col in transformed_df.columns if col.startswith(f'{pillar}_PC')]
    
    if not component_columns:
        print(f"No component scores available for {pillar}. Skipping clustering.")
        continue
    
    X_components = transformed_df[component_columns].values
    
    # Ensure we have enough data points
    if X_components.shape[0] < 3:
        print(f"Not enough data points for {pillar} to perform clustering. Skipping.")
        continue
        
    print(f"  Clustering based on {len(component_columns)} components...")
    
    # 1. Hierarchical Clustering (Ward's method)
    # Compute the linkage matrix
    Z = linkage(X_components, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(14, 8))
    plt.title(f'Hierarchical Clustering Dendrogram - {pillar}', fontsize=18)
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8., color_threshold=0.7*max(Z[:,2]))
    plt.xlabel('Communities (truncated)', fontsize=16)
    plt.ylabel('Distance', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/dendrogram_{pillar}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    max_clusters = min(10, X_components.shape[0] - 1)  # Cap at 10 clusters
    
    for n_clusters in range(2, max_clusters + 1):
        # K-means clustering with error handling
        try:
            # Try with default settings and explicit threadpool handling
            import os
            os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP threads to 1
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_components)
        except Exception as e:
            print(f"  Warning: Error during KMeans clustering: {e}")
            print(f"  Using alternative approach for KMeans")
            
            # Alternative approach - simplest possible KMeans implementation
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
            cluster_labels = kmeans.fit_predict(X_components)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_components, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"  For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-', color=pillar_colors.get(pillar, '#440154'))
    plt.title(f'Silhouette Scores for Clustering - {pillar}', fontsize=18)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Silhouette Score', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Step_4/output/figures/silhouette_scores_{pillar}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determine optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because index starts at 0 and range starts at 2
    print(f"  Optimal number of clusters for {pillar}: {optimal_clusters}")
    
    # 3. K-means with optimal number of clusters
    try:
        os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP threads to 1
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_components)
    except Exception as e:
        print(f"  Warning: Error during KMeans clustering: {e}")
        print(f"  Using MiniBatchKMeans as fallback")
        kmeans = MiniBatchKMeans(n_clusters=optimal_clusters, random_state=42, batch_size=100)
        kmeans_labels = kmeans.fit_predict(X_components)
    
    # 4. Hierarchical clustering with optimal cut
    hierarchical_labels = fcluster(Z, optimal_clusters, criterion='maxclust') - 1  # -1 to match 0-indexing
    
    # 5. Compute Rand index to assess stability between methods
    rand_index = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    print(f"  Adjusted Rand Index between k-means and hierarchical clustering: {rand_index:.3f}")
    
    # Store cluster assignments
    hierarchical_clusters[pillar] = hierarchical_labels
    kmeans_clusters[pillar] = kmeans_labels
    
    # Add cluster labels to the component scores dataframe
    transformed_df[f'{pillar}_Cluster_Hierarchical'] = hierarchical_labels
    transformed_df[f'{pillar}_Cluster_KMeans'] = kmeans_labels
    
    # 6. Visualize clusters (using 2 main components)
    if X_components.shape[1] >= 2:
        plt.figure(figsize=(20, 10))
        
        # Plot hierarchical clusters
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_components[:, 0], X_components[:, 1], 
                             c=hierarchical_labels, cmap='viridis', 
                             s=50, alpha=0.7, edgecolors='w')
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Hierarchical Clustering - {pillar}', fontsize=16)
        plt.xlabel(f'{pillar}_PC1', fontsize=14)
        plt.ylabel(f'{pillar}_PC2', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Plot k-means clusters
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_components[:, 0], X_components[:, 1], 
                             c=kmeans_labels, cmap='viridis', 
                             s=50, alpha=0.7, edgecolors='w')
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'K-means Clustering - {pillar}', fontsize=16)
        plt.xlabel(f'{pillar}_PC1', fontsize=14)
        plt.ylabel(f'{pillar}_PC2', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'Step_4/output/figures/clusters_visualization_{pillar}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Analyze stability between different runs
    # Run k-means 5 times with different random states
    stability_scores = []
    try:
        os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP threads to 1
        base_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        base_labels = base_kmeans.fit_predict(X_components)
        
        for seed in range(10, 15):
            test_kmeans = KMeans(n_clusters=optimal_clusters, random_state=seed, n_init=10)
            test_labels = test_kmeans.fit_predict(X_components)
            stability_score = adjusted_rand_score(base_labels, test_labels)
            stability_scores.append(stability_score)
    except Exception as e:
        print(f"  Warning: Error during stability analysis: {e}")
        print(f"  Using MiniBatchKMeans for stability analysis")
        
        base_kmeans = MiniBatchKMeans(n_clusters=optimal_clusters, random_state=42, batch_size=100)
        base_labels = base_kmeans.fit_predict(X_components)
        
        for seed in range(10, 15):
            test_kmeans = MiniBatchKMeans(n_clusters=optimal_clusters, random_state=seed, batch_size=100)
            test_labels = test_kmeans.fit_predict(X_components)
            stability_score = adjusted_rand_score(base_labels, test_labels)
            stability_scores.append(stability_score)
    
    avg_stability = sum(stability_scores) / len(stability_scores)
    print(f"  Average stability score (Rand index) across 5 runs: {avg_stability:.3f}")
    
    # Store results
    cluster_results[pillar] = {
        'optimal_clusters': optimal_clusters,
        'silhouette_scores': silhouette_scores,
        'hierarchical_labels': hierarchical_labels,
        'kmeans_labels': kmeans_labels,
        'rand_index': rand_index,
        'stability_score': avg_stability
    }

# Update the combined dataframe with cluster assignments
for pillar, transformed_df in transformed_data.items():
    for col in transformed_df.columns:
        if col.startswith(f'{pillar}_Cluster_'):
            all_components_df[col] = transformed_df[col]

# Save updated component scores with cluster assignments
all_components_df.to_csv('Step_4/data/all_component_scores_with_clusters.csv', index=False)
print("\nComponent scores with cluster assignments saved to 'Step_4/data/all_component_scores_with_clusters.csv'")

# Create a clustering summary table
clustering_summary_rows = []

for pillar, results in cluster_results.items():
    clustering_summary_rows.append({
        'Pillar': pillar,
        'Optimal_Clusters': results['optimal_clusters'],
        'Best_Silhouette_Score': max(results['silhouette_scores']),
        'Rand_Index': results['rand_index'],
        'Stability_Score': results['stability_score']
    })

clustering_summary_df = pd.DataFrame(clustering_summary_rows)
clustering_summary_df.to_csv('Step_4/data/clustering_summary.csv', index=False)
print("\nClustering summary saved to 'Step_4/data/clustering_summary.csv'") 