# Theoretical Approach to Variable Selection in Step 4

## Overview

This document explains the theoretical approach to variable selection that was implemented in Step 4 of our Community Crime-Risk Index (CCRI) project. 

## Background

In Step 4 of the multivariate analysis, we initially used purely statistical criteria to select indicators, such as:
- High communality (> 0.5)
- Low collinearity with other indicators (< 0.95)
- Strong representation in principal components

This approach, while statistically sound, led to the selection of only 8 indicators across the four pillars (Demographics, Income, Housing, Crime), with just 2 indicators per pillar. However, this was deemed insufficient for capturing the multidimensional nature of community crime risk as conceptualized in our theoretical framework.

## Implementation of Theoretical Approach

We modified our approach to prioritize theoretical importance over purely statistical considerations. We implemented this change through:

1. **Pillar-Specific Theoretical Framework**: We identified key variables for each pillar based on established criminological theories:
   - **Social Disorganization Theory** (Shaw & McKay, Sampson & Groves): Emphasizes social cohesion, neighborhood stability, and concentrated disadvantage
   - **Routine Activity Theory** (Cohen & Felson): Focuses on motivated offenders, suitable targets, and capable guardians

2. **Implementation Files**:
   - `implement_theoretical_selection.py`: The main script that implements the theoretical selection
   - `theoretical_analysis_summary.md`: Documentation of the theoretical approach and results
   - `theoretical_trimmed_dataset.csv`: The dataset with theoretically selected variables
   - `theoretical_final_indicators.csv`: Details on the selected indicators and their rationale

3. **Selection Results**:
   - **Demographics Pillar**: 4 indicators (racepctblack, racePctHisp, pctUrban, PctNotSpeakEnglWell)
   - **Income Pillar**: 4 indicators (PctPopUnderPov, medIncome, pctWPubAsst, PctUnemployed)
   - **Housing Pillar**: 5 indicators (PctHousOccup, PctSameHouse85, PctVacantBoarded, PctHousNoPhone, PctFam2Par)
   - **Crime Pillar**: 5 indicators (ViolentCrimesPerPop, murdPerPop, robbbPerPop, autoTheftPerPop, arsonsPerPop)

4. **Integration with Workflow**:
   - The theoretically selected dataset (`theoretical_trimmed_dataset.csv`) has replaced the statistically selected dataset (`step4_trimmed_dataset.csv`)
   - The main analysis summary has been updated to reflect the theoretical approach
   - The final indicators file has been updated to include all theoretically important variables

## Rationale for Change

The theoretical approach ensures:

1. **Theoretical Validity**: The composite index maintains fidelity to established criminological theories
2. **Balanced Representation**: Each pillar has adequate representation (4-5 indicators)
3. **Comprehensive Coverage**: All key dimensions of community crime risk are captured
4. **Future Analysis Support**: Provides sufficient variables for meaningful weighting and aggregation in Steps 5 and 6

## Next Steps

The theoretically selected indicators will proceed to:
1. **Normalization (Step 5)**: Variables will be normalized to comparable scales
2. **Weighting and Aggregation (Step 6)**: Variables will be weighted according to their theoretical importance

This theoretically-grounded approach will result in a more valid and meaningful Community Crime-Risk Index. 