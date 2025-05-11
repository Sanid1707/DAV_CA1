#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix indicators - Update final_indicators.csv to match with step4_trimmed_dataset.csv
"""

import pandas as pd
import os

print("Fixing final indicators based on trimmed dataset...")

# Load the trimmed dataset to identify which indicators were actually kept
trimmed_df = pd.read_csv('../output/step4_trimmed_dataset.csv')
actual_indicators = list(trimmed_df.columns)
actual_indicators.remove('communityname')  # Remove non-indicator column

# Load the final indicators dataframe
final_indicators_df = pd.read_csv('../output/final_indicators.csv')

# Define expected theoretically important variables for each pillar
theory_important = {
    'Demographics': ['racepctblack', 'racePctHisp', 'pctUrban', 'PctNotSpeakEnglWell'],
    'Income': ['PctPopUnderPov', 'medIncome', 'pctWPubAsst', 'PctUnemployed'],
    'Housing': ['PctHousOccup', 'PctSameHouse85', 'PctVacantBoarded', 'PctHousNoPhone', 'PctFam2Par'],
    'Crime': ['ViolentCrimesPerPop', 'murdPerPop', 'robbbPerPop', 'autoTheftPerPop', 'arsonsPerPop']
}

# Map columns in trimmed dataset to their respective pillars
column_to_pillar = {}
for pillar, indicators in theory_important.items():
    for indicator in indicators:
        column_to_pillar[indicator] = pillar

# Update the final indicators dataframe to match what's in step4_trimmed_dataset.csv
for _, row in final_indicators_df.iterrows():
    indicator = row['Indicator']
    pillar = row['Pillar']
    
    # Check if this indicator is in the trimmed dataset (it was kept)
    if indicator in actual_indicators:
        # It should be marked as 'Keep' with weight 1.0
        final_indicators_df.loc[final_indicators_df['Indicator'] == indicator, 'Status'] = 'Keep'
        final_indicators_df.loc[final_indicators_df['Indicator'] == indicator, 'Weight'] = 1.0
        
        # If it was theoretically important, update the rationale
        if indicator in theory_important.get(pillar, []):
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
            
            final_indicators_df.loc[final_indicators_df['Indicator'] == indicator, 'Rationale'] = rationale
    else:
        # It should be marked as 'Drop' with weight 0.0
        final_indicators_df.loc[final_indicators_df['Indicator'] == indicator, 'Status'] = 'Drop'
        final_indicators_df.loc[final_indicators_df['Indicator'] == indicator, 'Weight'] = 0.0

# Save updated final indicators
final_indicators_df.to_csv('../output/final_indicators.csv', index=False)
print(f"Updated final_indicators.csv saved.")

# Create count summary
keep_count = final_indicators_df[final_indicators_df['Status'] == 'Keep'].shape[0]
drop_count = final_indicators_df[final_indicators_df['Status'] == 'Drop'].shape[0]
print(f"Indicators kept: {keep_count}, Indicators dropped: {drop_count}")

# Show counts by pillar
pillar_counts = final_indicators_df[final_indicators_df['Status'] == 'Keep'].groupby('Pillar').size()
print("\nIndicators kept by pillar:")
for pillar, count in pillar_counts.items():
    print(f"  {pillar}: {count}")

print("\nTheoretically important indicators that were kept:")
for pillar, indicators in theory_important.items():
    kept = [ind for ind in indicators if ind in actual_indicators]
    print(f"  {pillar}: {', '.join(kept)}")

print("\nDone!") 