#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement Theoretical Selection - Create a new trimmed dataset with theoretically important variables
This script implements our theoretical approach to variable selection without modifying the original code.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.ticker as mtick

print("\n" + "="*80)
print("IMPLEMENTING THEORETICAL APPROACH TO VARIABLE SELECTION")
print("="*80)

# Define theoretically important variables for each pillar based on our framework
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

# Get all theoretical variables as a flat list
all_theoretical_vars = []
for pillar, vars_list in theoretical_importance.items():
    all_theoretical_vars.extend(vars_list)

# Load original dataset
try:
    df = pd.read_csv('../../step3_final_dataset_for_multivariate.csv')
    print(f"Loaded original dataset with {df.shape[0]} communities and {df.shape[1]} variables")
except FileNotFoundError:
    try:
        df = pd.read_csv('../step3_final_dataset_for_multivariate.csv')
        print(f"Loaded original dataset from alternative path")
    except:
        print("Error: Could not find the original dataset.")
        exit(1)

# Check which theoretical variables exist in the dataset
available_vars = [var for var in all_theoretical_vars if var in df.columns]
missing_vars = [var for var in all_theoretical_vars if var not in df.columns]

if missing_vars:
    print(f"Warning: {len(missing_vars)} theoretical variables not found in dataset: {', '.join(missing_vars)}")

print(f"Found {len(available_vars)} theoretically important variables in the dataset")

# Create directories if they don't exist
os.makedirs('../output/figures', exist_ok=True)

# Define function to run PCA on each pillar
def run_pca_on_pillar(df, pillar, variables):
    """Run PCA on a specific pillar's variables and visualize results"""
    print(f"\nAnalyzing {pillar} Pillar with {len(variables)} variables...")
    
    # Standardize the data
    X = df[variables].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=variables)
    
    # Run PCA
    pca = PCA()
    principal_components = pca.fit_transform(X_scaled_df)
    
    # Create scree plot
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
    plt.title(f'Scree Plot - {pillar} Pillar (Theoretical Selection)')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.ylim(0, 1.05)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'../output/figures/theoretical_scree_plot_{pillar}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create loadings heatmap
    loadings = pd.DataFrame(
        pca.components_.T,
        index=variables,
        columns=[f'PC{i+1}' for i in range(len(variables))]
    )
    
    return {
        'pca': pca,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }

# Analyze each pillar
pillar_results = {}
pillar_vars = {}

for pillar, theo_vars in theoretical_importance.items():
    # Get variables that exist in the dataset
    available_pillar_vars = [var for var in theo_vars if var in df.columns]
    pillar_vars[pillar] = available_pillar_vars
    
    if len(available_pillar_vars) >= 2:  # Need at least 2 variables for PCA
        results = run_pca_on_pillar(df, pillar, available_pillar_vars)
        pillar_results[pillar] = results
    else:
        print(f"Warning: Not enough variables for {pillar} pillar. Skipping PCA.")

# Create new trimmed dataset with theoretical variables
trimmed_vars = ['communityname'] if 'communityname' in df.columns else []
for pillar, vars_list in pillar_vars.items():
    trimmed_vars.extend(vars_list)

trimmed_df = df[trimmed_vars].copy()

# Save the new theoretically-driven dataset
trimmed_df.to_csv('../output/theoretical_trimmed_dataset.csv', index=False)
print(f"\nSaved theoretical trimmed dataset with {len(trimmed_vars)-1} variables to 'theoretical_trimmed_dataset.csv'")

# Create a final indicators CSV
final_indicators = []

for pillar, variables in theoretical_importance.items():
    for indicator in variables:
        if indicator in df.columns:
            # Determine if this indicator is in our pillars and available
            is_kept = indicator in pillar_vars.get(pillar, [])
            
            # Create descriptive rationale
            if is_kept:
                rationale = "Theoretically important variable for our crime risk framework based on "
                
                if pillar == 'Demographics':
                    if indicator == 'racepctblack' or indicator == 'racePctHisp':
                        rationale += "Social Disorganization Theory - racial/ethnic heterogeneity."
                    elif indicator == 'pctUrban':
                        rationale += "Shaw & McKay's work on urbanization and crime."
                    elif indicator == 'PctNotSpeakEnglWell':
                        rationale += "social isolation factors that may reduce collective efficacy."
                
                elif pillar == 'Income':
                    if indicator == 'PctPopUnderPov':
                        rationale += "concentrated disadvantage as a key predictor in Social Disorganization Theory."
                    elif indicator == 'medIncome':
                        rationale += "economic capacity for guardianship in Routine Activity Theory."
                    elif indicator == 'pctWPubAsst':
                        rationale += "economic vulnerability and dependency markers."
                    elif indicator == 'PctUnemployed':
                        rationale += "unemployment as a driver of motivated offenders in Routine Activity Theory."
                
                elif pillar == 'Housing':
                    if indicator == 'PctHousOccup':
                        rationale += "occupancy as a deterrent to crime opportunities."
                    elif indicator == 'PctSameHouse85':
                        rationale += "residential stability/turnover central to Social Disorganization Theory."
                    elif indicator == 'PctVacantBoarded':
                        rationale += "physical disorder as a criminogenic factor."
                    elif indicator == 'PctHousNoPhone':
                        rationale += "reduced guardianship capability (ability to call for help)."
                    elif indicator == 'PctFam2Par':
                        rationale += "informal social control mechanisms via family structure."
                
                elif pillar == 'Crime':
                    if indicator == 'ViolentCrimesPerPop':
                        rationale += "comprehensive violent crime measure as a key outcome variable."
                    elif indicator == 'murdPerPop':
                        rationale += "murder as the most serious violent crime indicator."
                    elif indicator == 'robbbPerPop':
                        rationale += "robbery as a crime that spans both property and violent domains."
                    elif indicator == 'autoTheftPerPop':
                        rationale += "auto theft as a key property crime indicator."
                    elif indicator == 'arsonsPerPop':
                        rationale += "arson as an indicator of destructive property crime."
            else:
                rationale = "Variable not found in dataset"
                
            # Add to final indicators list
            final_indicators.append({
                'Pillar': pillar,
                'Indicator': indicator,
                'Importance_Score': 1.0 if is_kept else 0.0,  # High importance for theoretical variables
                'Status': 'Keep' if is_kept else 'Missing',
                'Rationale': rationale,
                'Weight': 1.0 if is_kept else 0.0
            })

# Create final indicators DataFrame and save it
final_indicators_df = pd.DataFrame(final_indicators)
final_indicators_df.to_csv('../output/theoretical_final_indicators.csv', index=False)
print(f"Saved theoretical final indicators to 'theoretical_final_indicators.csv'")

# Generate a summary of our analysis
with open('../output/theoretical_analysis_summary.md', 'w') as f:
    f.write("# Theoretical Approach to Variable Selection Summary\n\n")
    
    f.write("## Overview\n\n")
    f.write("This document summarizes the theoretical approach to variable selection for our Community Crime-Risk Index (CCRI).\n")
    f.write("Rather than relying solely on statistical criteria, we've prioritized variables that are theoretically important based on established criminological theories.\n\n")
    
    f.write("## Selected Variables by Pillar\n\n")
    
    for pillar, variables in pillar_vars.items():
        f.write(f"### {pillar} Pillar\n\n")
        f.write(f"- Selected indicators: {len(variables)}\n")
        f.write(f"- Indicators: {', '.join(variables)}\n")
        
        if pillar in pillar_results:
            results = pillar_results[pillar]
            var_pc1 = results['explained_variance'][0] if len(results['explained_variance']) > 0 else 0
            var_total = sum(results['explained_variance'])
            f.write(f"- Variance explained by PC1: {var_pc1:.2f} ({var_pc1*100:.1f}%)\n")
            f.write(f"- Total variance explained by all components: {var_total:.2f} ({var_total*100:.1f}%)\n")
        
        f.write("\n**Theoretical Rationale:**\n\n")
        
        for var in variables:
            rationale = ""
            if pillar == 'Demographics':
                if var == 'racepctblack' or var == 'racePctHisp':
                    rationale = "Measure of racial/ethnic heterogeneity, a key factor in Social Disorganization Theory."
                elif var == 'pctUrban':
                    rationale = "Urbanization is central to Shaw & McKay's work on spatial distribution of crime."
                elif var == 'PctNotSpeakEnglWell':
                    rationale = "Indicator of social isolation that may reduce collective efficacy and community cohesion."
            
            elif pillar == 'Income':
                if var == 'PctPopUnderPov':
                    rationale = "Concentrated disadvantage is a fundamental predictor in Social Disorganization Theory."
                elif var == 'medIncome':
                    rationale = "Economic capacity affects guardianship capabilities in Routine Activity Theory."
                elif var == 'pctWPubAsst':
                    rationale = "Public assistance dependency reflects economic vulnerability of communities."
                elif var == 'PctUnemployed':
                    rationale = "Unemployment is associated with motivated offenders in Routine Activity Theory."
            
            elif pillar == 'Housing':
                if var == 'PctHousOccup':
                    rationale = "Occupied housing serves as a deterrent to crime opportunities."
                elif var == 'PctSameHouse85':
                    rationale = "Residential stability is inversely related to crime in Social Disorganization Theory."
                elif var == 'PctVacantBoarded':
                    rationale = "Physical disorder signals low social control and creates crime opportunities."
                elif var == 'PctHousNoPhone':
                    rationale = "Lack of phone service reduces guardianship capability (ability to call for help)."
                elif var == 'PctFam2Par':
                    rationale = "Two-parent families represent informal social control mechanisms in communities."
            
            elif pillar == 'Crime':
                if var == 'ViolentCrimesPerPop':
                    rationale = "Comprehensive measure of violent crime, capturing overall community violence."
                elif var == 'murdPerPop':
                    rationale = "Murder rate indicates the most serious violent crime in a community."
                elif var == 'robbbPerPop':
                    rationale = "Robbery spans both property and violent domains, indicating motivated offenders."
                elif var == 'autoTheftPerPop':
                    rationale = "Auto theft is a key property crime indicator involving valuable targets."
                elif var == 'arsonsPerPop':
                    rationale = "Arson represents destructive property crime and community disorder."
            
            f.write(f"- **{var}**: {rationale}\n")
        
        f.write("\n")
    
    f.write("## Implications for Analysis\n\n")
    f.write("Our theoretical approach has the following advantages:\n\n")
    f.write("1. **Theoretical Coherence**: Ensures our composite index aligns with established criminological theories\n")
    f.write("2. **Comprehensive Coverage**: Includes all key dimensions of community crime risk even when statistical metrics might suggest removal\n")
    f.write("3. **Interpretability**: Makes the final composite index more meaningful to criminologists and policy makers\n")
    f.write("4. **Balanced Representation**: Ensures all pillars have sufficient representation in the final index\n\n")
    
    f.write("## Next Steps\n\n")
    f.write("The theoretical variables selected here will proceed to:\n\n")
    f.write("1. **Normalization (Step 5)**: Variables will be normalized to comparable scales\n")
    f.write("2. **Weighting and Aggregation (Step 6)**: Variables will be weighted according to their theoretical importance\n\n")
    
    f.write("This theoretically-grounded approach will result in a more valid and meaningful Community Crime-Risk Index.")

print(f"Generated theoretical analysis summary in 'theoretical_analysis_summary.md'")
print("\nTheoretical selection implementation complete!") 