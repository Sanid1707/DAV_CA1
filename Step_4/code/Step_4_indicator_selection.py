#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Multivariate Analysis - Sections G-I
- Create decision grid for indicator selection
- Re-run PCA on trimmed indicator set
- Document results and prepare for Step 5
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from collections import Counter
import matplotlib.ticker as mtick
import scipy.stats as stats
import warnings
import re

# Create directories if they don't exist
os.makedirs('Step_4/output/figures', exist_ok=True)

print("\n" + "="*80)
print("SECTIONS 4-G to 4-I: INDICATOR SELECTION, PCA RERUN, AND DOCUMENTATION")
print("="*80)

# Load data
try:
    # Load the original data
    df = pd.read_csv('../../step3_final_dataset_for_multivariate.csv')
    print(f"Loaded original data with {df.shape[0]} communities and {df.shape[1]} variables")
    
    # Load indicator decision grid
    indicator_decisions = pd.read_csv('../output/indicator_decision_grid.csv')
    print(f"Loaded indicator decision grid with {indicator_decisions.shape[0]} indicators")
    
except FileNotFoundError:
    try:
        # Alternative paths - try different relative paths
        df = pd.read_csv('../step3_final_dataset_for_multivariate.csv')
        print(f"Loaded data from alternative path 1")
        indicator_decisions = pd.read_csv('../output/indicator_decision_grid.csv')
        print(f"Loaded indicator decision grid from alternative path")
    except FileNotFoundError:
        try:
            # Try another alternative path
            df = pd.read_csv('step3_final_dataset_for_multivariate.csv')
            print(f"Loaded data from alternative path 2")
            indicator_decisions = pd.read_csv('Step_4/output/indicator_decision_grid.csv')
            print(f"Loaded indicator decision grid from alternative path 2")
        except:
            print("Error: Could not load required data files")
            print("Please run Step_4_collinearity.py first to generate the indicator decision grid")
            exit(1)

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

# Define theoretically important variables for each pillar based on our theoretical framework
# These variables will be kept regardless of their statistical properties
theoretical_importance = {
    'Demographics': [
        'racepctblack',      # Important for measuring racial composition related to social disorganization
        'racePctHisp',       # Important for ethnic heterogeneity assessment
        'pctUrban',          # Central to Shaw & McKay's work on urbanization and crime
        'PctNotSpeakEnglWell' # Indicator of social isolation that may reduce collective efficacy
    ],
    'Income': [
        'PctPopUnderPov',    # Fundamental measure of concentrated disadvantage
        'medIncome',         # Core socioeconomic indicator affecting guardian capability
        'pctWPubAsst',       # Marker of economic vulnerability and dependency
        'PctUnemployed'      # Key predictor of motivated offenders in routine activity theory
    ],
    'Housing': [
        'PctHousOccup',      # Indicator of neighborhood stability and vacant property presence
        'PctSameHouse85',    # Direct measure of residential stability/turnover
        'PctVacantBoarded',  # Strong indicator of physical disorder
        'PctHousNoPhone',    # Measure of guardianship capability (ability to call for help)
        'PctFam2Par'         # Two-parent families as informal social control mechanism
    ],
    'Crime': [
        'ViolentCrimesPerPop', # Comprehensive violent crime measure
        'murdPerPop',         # Most serious violent crime indicator
        'robbbPerPop',        # Property crime with violence element
        'autoTheftPerPop',    # Property crime indicator (vehicle theft)
        'arsonsPerPop'        # Property destruction indicator
    ]
}

# 4-G: Final decision grid for indicators
print("\n4-G: Creating final decision grid for indicators...")

# Create the final indicator selection
final_indicators = []

# Define minimum number of indicators to keep per pillar
# Increased from 2 to 4 to accommodate more theoretical variables
MIN_INDICATORS_PER_PILLAR = 4

# Function to calculate indicator importance score with increased emphasis on theoretical importance
def calculate_importance_score(row, pillar_decisions, pillar, indicator):
    """
    Calculate a balanced importance score based on multiple criteria
    Returns a score between 0 and 1, where higher is more important
    Now with increased weight for theoretical importance
    """
    # Statistical quality (communality)
    communality = row['Communality'] if 'Communality' in row else 0.5
    
    # Uniqueness (inverse of being in high correlation group)
    # Note: we're now using a correlation threshold of 0.95 instead of 0.9
    uniqueness = 0.0 if row['In_High_Correlation_Group'] else 1.0
    
    # PC Representation - check if it's the top loading on any PC
    is_top_loading = False
    if 'PCA Top Loading' in row:
        is_top_loading = row['PCA Top Loading']
    else:
        # Try to infer from rationale or decision
        is_top_loading = 'representative' in row['Decision'].lower() or 'highest communality' in row['Rationale'].lower()
    
    pc_representation = 1.0 if is_top_loading else 0.5
    
    # Conceptual importance (now checking against our theoretical framework)
    theoretical_value = 1.0 if indicator in theoretical_importance.get(pillar, []) else 0.5
    
    # Calculate weighted score with increased weight on theoretical importance
    score = (
        0.20 * communality +         # Statistical quality
        0.15 * uniqueness +          # Uniqueness of information
        0.15 * pc_representation +   # Representation of principal components
        0.50 * theoretical_value     # Conceptual importance (increased weight)
    )
    
    return score

for pillar, variables in pillars.items():
    print(f"\nFinalizing indicators for {pillar} pillar...")
    
    # Get decisions for this pillar
    pillar_decisions = indicator_decisions[indicator_decisions['Pillar'] == pillar]
    
    if pillar_decisions.empty:
        print(f"  No decisions found for {pillar}. Keeping all variables.")
        for var in variables:
            if var in df.columns:
                final_indicators.append({
                    'Pillar': pillar,
                    'Indicator': var,
                    'Status': 'Keep',
                    'Rationale': 'No multivariate analysis available, keeping by default',
                    'Weight': 1.0  # Equal weights for now, to be adjusted in Step 6
                })
        continue
    
    # Initial classification based on decision
    keep_indicators = []
    consider_indicators = []
    drop_indicators = []
    
    # Calculate importance scores for all indicators
    importance_scores = {}
    for _, row in pillar_decisions.iterrows():
        indicator = row['Indicator']
        importance_scores[indicator] = calculate_importance_score(row, pillar_decisions, pillar, indicator)
        
        if row['Decision'].startswith('Keep'):
            keep_indicators.append(indicator)
        elif 'merging' in row['Decision']:
            consider_indicators.append(indicator)
        elif row['Decision'].startswith('Drop'):
            drop_indicators.append(indicator)
    
    # Sort indicators by importance score
    sorted_indicators = sorted(
        [(ind, importance_scores[ind]) for ind in variables if ind in importance_scores],
        key=lambda x: x[1],
        reverse=True  # Higher scores first
    )
    
    # First, automatically include all theoretically important variables
    final_keep_indicators = []
    for indicator in theoretical_importance.get(pillar, []):
        if indicator in variables and indicator in importance_scores:
            final_keep_indicators.append(indicator)
    
    # Add indicators previously marked as "keep" if not already included
    for indicator in keep_indicators:
        if indicator not in final_keep_indicators:
            final_keep_indicators.append(indicator)
    
    # If we still don't have enough, add from consider_indicators by importance score
    if len(final_keep_indicators) < MIN_INDICATORS_PER_PILLAR:
        sorted_consider = sorted(
            [(ind, importance_scores[ind]) for ind in consider_indicators],
            key=lambda x: x[1],
            reverse=True
        )
        
        for ind, _ in sorted_consider:
            if ind not in final_keep_indicators:
                final_keep_indicators.append(ind)
                if len(final_keep_indicators) >= MIN_INDICATORS_PER_PILLAR:
                    break
    
    # If still not enough, add highest scoring dropped indicators
    if len(final_keep_indicators) < MIN_INDICATORS_PER_PILLAR:
        sorted_drop = sorted(
            [(ind, importance_scores[ind]) for ind in drop_indicators],
            key=lambda x: x[1],
            reverse=True
        )
        
        for ind, _ in sorted_drop:
            if ind not in final_keep_indicators:
                final_keep_indicators.append(ind)
                if len(final_keep_indicators) >= MIN_INDICATORS_PER_PILLAR:
                    break
    
    # Count final decisions
    keep_count = len(final_keep_indicators)
    drop_count = len([ind for ind in variables if ind in importance_scores and ind not in final_keep_indicators])
    
    print(f"  Relaxed theoretical selection: Keep {keep_count}, Drop {drop_count}")
    print(f"  Indicators to keep: {', '.join(final_keep_indicators)}")
    
    # Add final decisions to the indicator list
    for _, row in pillar_decisions.iterrows():
        indicator = row['Indicator']
        decision = row['Decision']
        rationale = row['Rationale']
        
        # Determine final status based on our selection
        if indicator in final_keep_indicators:
            status = 'Keep'
            weight = 1.0
            
            # Add explanation based on the reason for keeping
            if indicator in theoretical_importance.get(pillar, []):
                if decision.startswith('Drop'):
                    rationale = "Theoretically important variable for our framework despite " + rationale.lower()
                else:
                    rationale += ' Also identified as theoretically important for our framework.'
            # If it was originally marked for dropping but we're keeping it
            elif decision.startswith('Drop'):
                rationale += ' Kept to ensure minimum pillar representation and balanced selection.'
            elif 'merging' in decision:
                rationale += ' Kept based on balanced scoring across statistical and conceptual criteria.'
        else:
            status = 'Drop'
            weight = 0.0
            
            # If it was originally marked for keeping, add explanation
            if decision.startswith('Keep'):
                rationale += ' Dropped due to sufficient representation from other indicators in this pillar.'
        
        # Add indicator with updated status to final list
        final_indicators.append({
            'Pillar': pillar,
            'Indicator': indicator,
            'Importance_Score': importance_scores[indicator],
            'Status': status,
            'Rationale': rationale,
            'Weight': weight
        })

# Save final indicators to CSV
final_indicators_df = pd.DataFrame(final_indicators)
final_indicators_df.to_csv('Step_4/output/final_indicators.csv', index=False)
print(f"Saved final indicators to Step_4/output/final_indicators.csv")

# Count how many indicators we're keeping per pillar
keep_counts = final_indicators_df[final_indicators_df['Status'] == 'Keep'].groupby('Pillar').size()
print("\nFinal indicator counts per pillar:")
for pillar, count in keep_counts.items():
    print(f"  {pillar}: {count}")

# Get all indicators we're keeping
kept_indicators = final_indicators_df[final_indicators_df['Status'] == 'Keep']['Indicator'].tolist()
print(f"\nTotal indicators kept: {len(kept_indicators)}")
print("Kept indicators:")
for pillar in pillars.keys():
    pillar_indicators = final_indicators_df[(final_indicators_df['Pillar'] == pillar) & (final_indicators_df['Status'] == 'Keep')]['Indicator'].tolist()
    if pillar_indicators:
        print(f"  {pillar}: {', '.join(pillar_indicators)}")

# 4-H to 4-I: Re-running PCA on trimmed dataset and documenting results

# Helper functions 
def standardize_data(df, variables):
    """Standardize data for PCA."""
    X = df[variables].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=variables, index=df.index)

def run_pca(df, variables, n_components=None):
    """Run PCA on the given variables."""
    X = standardize_data(df, variables)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    # Create dataframe with principal components
    component_names = [f"PC{i+1}" for i in range(pca.n_components_)]
    pc_df = pd.DataFrame(data=principal_components, columns=component_names, index=df.index)
    
    # Add the original community name if available
    if 'communityname' in df.columns:
        pc_df['communityname'] = df['communityname']
    
    return pc_df, pca

def create_explained_variance_plot(pca, title=""):
    """Create scree plot for explained variance."""
    plt.figure(figsize=(10, 6))
    
    # Individual explained variance
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        label='Individual explained variance'
    )
    
    # Cumulative explained variance
    plt.step(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        where='mid',
        label='Cumulative explained variance',
        color='red'
    )
    
    plt.axhline(y=0.7, color='k', linestyle='--', alpha=0.3, label='70% explained variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Scree Plot - {title}')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.ylim(0, 1.05)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    return plt

print("\n4-H,I: Re-running PCA on trimmed indicator set and documenting results...")

# Create trimmed dataset with only the kept indicators
kept_vars = ['communityname'] + kept_indicators
trimmed_df = df[kept_vars].copy()

# Save the trimmed dataset
trimmed_df.to_csv('Step_4/output/step4_trimmed_dataset.csv', index=False)
print(f"Saved trimmed dataset with {trimmed_df.shape[1]-1} indicators to Step_4/output/step4_trimmed_dataset.csv")

# Re-run PCA on each pillar with the trimmed indicators
pillar_pca_results = {}
pillar_vars = {}

for pillar in pillars.keys():
    pillar_indicators = final_indicators_df[(final_indicators_df['Pillar'] == pillar) & (final_indicators_df['Status'] == 'Keep')]['Indicator'].tolist()
    if not pillar_indicators:
        continue
    
    pillar_vars[pillar] = pillar_indicators
    
    # Run PCA on this pillar
    pc_df, pca = run_pca(df, pillar_indicators)
    pillar_pca_results[pillar] = {
        'pc_df': pc_df,
        'pca': pca,
        'n_components': len(pillar_indicators),
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'loadings': pca.components_
    }
    
    # Create scree plot for this pillar
    plt = create_explained_variance_plot(pca, title=f"{pillar} Pillar - {len(pillar_indicators)} indicators")
    plt.savefig(f"Step_4/output/figures/scree_plot_{pillar}_trimmed.png", dpi=300, bbox_inches='tight')
    plt.close()

# Generate final analysis summary document
with open('Step_4/output/step4_analysis_summary.md', 'w') as f:
    f.write("\n# Step 4: Multivariate Analysis Summary\n\n")
    
    f.write("## Overview\n\n")
    f.write("This document summarizes the multivariate analysis conducted in Step 4, including:\n")
    f.write("- Principal Component Analysis (PCA) of indicators within each pillar\n")
    f.write("- Cluster analysis of communities\n")
    f.write("- Indicator collinearity assessment\n")
    f.write("- Final indicator selection\n\n")
    
    f.write("## Key Findings\n\n")
    
    f.write("### Principal Component Analysis\n\n")
    
    for pillar, results in pillar_pca_results.items():
        f.write(f"#### {pillar} Pillar\n\n")
        original_count = len(pillars[pillar])
        final_count = len(pillar_vars[pillar])
        
        f.write(f"- Original indicators: {original_count}\n")
        f.write(f"- Final indicators: {final_count}\n")
        
        if results['explained_variance'].size > 0:
            var_pc1 = results['explained_variance'][0]
            var_total = np.sum(results['explained_variance'])
            f.write(f"- Variance explained by PC1: {var_pc1:.2f} ({var_pc1*100:.1f}%)\n")
            f.write(f"- Total variance explained: {var_total:.2f} ({var_total*100:.1f}%)\n")
        
        f.write(f"- Final indicators: {', '.join(pillar_vars[pillar])}\n\n")
    
    f.write("\n### Cluster Analysis\n\n")
    f.write("Cluster analysis was performed using both hierarchical and k-means clustering methods on the principal component scores. The optimal number of clusters was determined using silhouette scores.\n\n")
    f.write("Detailed cluster analysis information not available.\n\n")
    
    f.write("\n### Indicator Selection\n\n")
    f.write("The following summarizes the final indicator selection based on multivariate analysis:\n\n")
    
    for pillar in pillars.keys():
        f.write(f"#### {pillar} Pillar\n\n")
        
        kept = final_indicators_df[(final_indicators_df['Pillar'] == pillar) & (final_indicators_df['Status'] == 'Keep')]['Indicator'].tolist()
        dropped = final_indicators_df[(final_indicators_df['Pillar'] == pillar) & (final_indicators_df['Status'] == 'Drop')]['Indicator'].tolist()
        
        f.write(f"- Selected indicators: {len(kept)}\n")
        f.write(f"- Indicators: {', '.join(kept)}\n")
        f.write(f"- Dropped indicators: {', '.join(dropped)}\n\n")
    
    f.write("\n### Theoretical Framework Considerations\n\n")
    f.write("This analysis has been specifically adjusted to prioritize variables that are theoretically important for our Community Crime-Risk Index (CCRI) framework. We have:\n\n")
    f.write("1. Prioritized variables central to Social Disorganization Theory (Shaw & McKay) and Routine Activity Theory\n")
    f.write("2. Relaxed the statistical criteria (such as communality thresholds) for theoretically important variables\n")
    f.write("3. Retained more variables per pillar to ensure comprehensive coverage of our theoretical constructs\n")
    f.write("4. Balanced statistical qualities with theoretical importance using a weighted scoring approach\n\n")
    
    f.write("This modified approach ensures that our final indicator set captures the multidimensional nature of community crime risk as conceptualized in our theoretical framework, while still maintaining statistical validity.\n\n")
    
    f.write("\n### Implications for Next Steps\n\n")
    f.write("The final set of indicators identified through multivariate analysis will be:\n")
    f.write("1. Re-normalized in Step 5\n")
    f.write("2. Weighted and aggregated in Step 6\n\n")
    
    f.write("This ensures that the composite indicators will be built on statistically sound foundations with:\n")
    f.write("- Reduced redundancy between indicators\n")
    f.write("- Balanced representation across conceptual dimensions\n")
    f.write("- Enhanced interpretability of results\n")
    f.write("- Strong theoretical grounding in criminological theory\n")

print("\nStep 4 indicator selection and documentation complete.\nUpdated final indicators are ready for Step 5 normalization.")

# Document the Approach to Indicator Selection
with open('Step_4/output/theoretical_selection_approach.md', 'w') as f:
    f.write("# Theoretical Approach to Variable Selection in Step 4\n\n")
    
    f.write("## Introduction\n\n")
    f.write("This document explains our modified approach to variable selection in Step 4, which prioritizes theoretical relevance while still considering statistical properties.\n\n")
    
    f.write("## Rationale for the Modified Approach\n\n")
    f.write("While traditional multivariate analysis often relies heavily on statistical criteria like communality and collinearity to select variables, we have modified our approach for the following reasons:\n\n")
    
    f.write("1. **Theoretical Framework Fidelity**: Our Community Crime-Risk Index (CCRI) is grounded in established criminological theories, particularly Social Disorganization Theory (Shaw & McKay) and Routine Activity Theory. Statistical criteria alone may eliminate variables that are conceptually central to these theories.\n\n")
    
    f.write("2. **Balanced Representation**: We need adequate representation from all four pillars (Demographics, Income, Housing, Crime) to create a theoretically sound composite index.\n\n")
    
    f.write("3. **Future Normalization and Weighting**: For Step 5 (normalization) and Step 6 (weighting), we need a more comprehensive set of variables to ensure our index captures the multidimensional nature of community crime risk.\n\n")
    
    f.write("4. **Avoiding Oversimplification**: Reducing our dataset too aggressively based solely on statistical criteria risks oversimplifying the complex phenomenon we're studying.\n\n")
    
    f.write("## Methodology Changes\n\n")
    f.write("We've implemented the following changes to our variable selection process:\n\n")
    
    f.write("1. **Increased Minimum Variables Per Pillar**: We've increased the minimum number of indicators per pillar from 2 to 4.\n\n")
    
    f.write("2. **Identification of Theoretically Important Variables**: We've explicitly identified variables that are theoretically important for each pillar, based on our research framework.\n\n")
    
    f.write("3. **Modified Importance Score Calculation**: We've adjusted our importance scoring algorithm to give greater weight (50%) to theoretical importance, while still considering statistical properties like communality (20%), uniqueness (15%), and PC representation (15%).\n\n")
    
    f.write("4. **Retention of Correlated Variables**: We've relaxed our stance on multicollinearity, allowing the retention of theoretically important variables even when they are highly correlated with others.\n\n")
    
    f.write("## Implications for the Analysis\n\n")
    f.write("This approach results in retaining more variables than a purely statistical selection would recommend. However, it ensures that our composite index:\n\n")
    
    f.write("1. Maintains strong theoretical validity\n")
    f.write("2. Captures all key dimensions of our research framework\n")
    f.write("3. Provides sufficient variables for meaningful weighting and aggregation in later steps\n")
    f.write("4. Represents each pillar adequately\n\n")
    
    f.write("While this approach may introduce some statistical redundancy, the benefits of theoretical coherence and comprehensive coverage outweigh these concerns for our specific research objectives.\n")

print("Created documentation on theoretical selection approach in Step_4/output/theoretical_selection_approach.md")

print("\nStep 4 indicator selection and documentation complete.") 