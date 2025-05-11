# Theoretical Approach to Variable Selection Summary

## Overview

This document summarizes the theoretical approach to variable selection for our Community Crime-Risk Index (CCRI).
Rather than relying solely on statistical criteria, we've prioritized variables that are theoretically important based on established criminological theories.

## Selected Variables by Pillar

### Demographics Pillar

- Selected indicators: 4
- Indicators: racepctblack, racePctHisp, pctUrban, PctNotSpeakEnglWell
- Variance explained by PC1: 0.48 (47.7%)
- Total variance explained by all components: 1.00 (100.0%)

**Theoretical Rationale:**

- **racepctblack**: Measure of racial/ethnic heterogeneity, a key factor in Social Disorganization Theory.
- **racePctHisp**: Measure of racial/ethnic heterogeneity, a key factor in Social Disorganization Theory.
- **pctUrban**: Urbanization is central to Shaw & McKay's work on spatial distribution of crime.
- **PctNotSpeakEnglWell**: Indicator of social isolation that may reduce collective efficacy and community cohesion.

### Income Pillar

- Selected indicators: 4
- Indicators: PctPopUnderPov, medIncome, pctWPubAsst, PctUnemployed
- Variance explained by PC1: 0.80 (80.4%)
- Total variance explained by all components: 1.00 (100.0%)

**Theoretical Rationale:**

- **PctPopUnderPov**: Concentrated disadvantage is a fundamental predictor in Social Disorganization Theory.
- **medIncome**: Economic capacity affects guardianship capabilities in Routine Activity Theory.
- **pctWPubAsst**: Public assistance dependency reflects economic vulnerability of communities.
- **PctUnemployed**: Unemployment is associated with motivated offenders in Routine Activity Theory.

### Housing Pillar

- Selected indicators: 5
- Indicators: PctHousOccup, PctSameHouse85, PctVacantBoarded, PctHousNoPhone, PctFam2Par
- Variance explained by PC1: 0.47 (47.0%)
- Total variance explained by all components: 1.00 (100.0%)

**Theoretical Rationale:**

- **PctHousOccup**: Occupied housing serves as a deterrent to crime opportunities.
- **PctSameHouse85**: Residential stability is inversely related to crime in Social Disorganization Theory.
- **PctVacantBoarded**: Physical disorder signals low social control and creates crime opportunities.
- **PctHousNoPhone**: Lack of phone service reduces guardianship capability (ability to call for help).
- **PctFam2Par**: Two-parent families represent informal social control mechanisms in communities.

### Crime Pillar

- Selected indicators: 5
- Indicators: ViolentCrimesPerPop, murdPerPop, robbbPerPop, autoTheftPerPop, arsonsPerPop
- Variance explained by PC1: 0.66 (66.2%)
- Total variance explained by all components: 1.00 (100.0%)

**Theoretical Rationale:**

- **ViolentCrimesPerPop**: Comprehensive measure of violent crime, capturing overall community violence.
- **murdPerPop**: Murder rate indicates the most serious violent crime in a community.
- **robbbPerPop**: Robbery spans both property and violent domains, indicating motivated offenders.
- **autoTheftPerPop**: Auto theft is a key property crime indicator involving valuable targets.
- **arsonsPerPop**: Arson represents destructive property crime and community disorder.

## Implications for Analysis

Our theoretical approach has the following advantages:

1. **Theoretical Coherence**: Ensures our composite index aligns with established criminological theories
2. **Comprehensive Coverage**: Includes all key dimensions of community crime risk even when statistical metrics might suggest removal
3. **Interpretability**: Makes the final composite index more meaningful to criminologists and policy makers
4. **Balanced Representation**: Ensures all pillars have sufficient representation in the final index

## Next Steps

The theoretical variables selected here will proceed to:

1. **Normalization (Step 5)**: Variables will be normalized to comparable scales
2. **Weighting and Aggregation (Step 6)**: Variables will be weighted according to their theoretical importance

This theoretically-grounded approach will result in a more valid and meaningful Community Crime-Risk Index.