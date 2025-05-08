
# Step 4: Multivariate Analysis Summary

## Overview

This document summarizes the multivariate analysis conducted in Step 4, including:
- Principal Component Analysis (PCA) of indicators within each pillar
- Cluster analysis of communities
- Indicator collinearity assessment
- Final indicator selection

## Key Findings

### Principal Component Analysis

#### Demographics Pillar

- Original indicators: 9
- Final indicators: 4
- Variance explained by PC1: 0.48 (47.7%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: racepctblack, racePctHisp, pctUrban, PctNotSpeakEnglWell

#### Income Pillar

- Original indicators: 5
- Final indicators: 4
- Variance explained by PC1: 0.80 (80.4%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: medIncome, pctWPubAsst, PctPopUnderPov, PctUnemployed

#### Housing Pillar

- Original indicators: 14
- Final indicators: 5
- Variance explained by PC1: 0.47 (47.0%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: PctFam2Par, PctHousOccup, PctVacantBoarded, PctHousNoPhone, PctSameHouse85

#### Crime Pillar

- Original indicators: 18
- Final indicators: 5
- Variance explained by PC1: 0.66 (66.2%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: murdPerPop, robbbPerPop, autoTheftPerPop, arsonsPerPop, ViolentCrimesPerPop


### Cluster Analysis

Cluster analysis was performed using both hierarchical and k-means clustering methods on the principal component scores. The optimal number of clusters was determined using silhouette scores.

Detailed cluster analysis information not available.


### Indicator Selection

The following summarizes the final indicator selection based on multivariate analysis:

#### Demographics Pillar

- Selected indicators: 4
- Indicators: racepctblack, racePctHisp, pctUrban, PctNotSpeakEnglWell
- Dropped indicators: population, racePctWhite, racePctAsian, PctImmigRec5, PctRecentImmig

#### Income Pillar

- Selected indicators: 4
- Indicators: medIncome, pctWPubAsst, PctPopUnderPov, PctUnemployed
- Dropped indicators: pctWSocSec

#### Housing Pillar

- Selected indicators: 5
- Indicators: PctFam2Par, PctHousOccup, PctVacantBoarded, PctHousNoPhone, PctSameHouse85
- Dropped indicators: PersPerFam, PctLargHouseFam, PersPerRentOccHous, PctPersDenseHous, PctWOFullPlumb, OwnOccQrange, RentQrange, NumInShelters, PctUsePubTrans

#### Crime Pillar

- Selected indicators: 5
- Indicators: murdPerPop, robbbPerPop, autoTheftPerPop, arsonsPerPop, ViolentCrimesPerPop
- Dropped indicators: murders, rapes, rapesPerPop, robberies, assaults, assaultPerPop, burglaries, burglPerPop, larcenies, larcPerPop, autoTheft, arsons, nonViolPerPop


### Theoretical Framework Considerations

This analysis has been specifically adjusted to prioritize variables that are theoretically important for our Community Crime-Risk Index (CCRI) framework. We have:

1. Prioritized variables central to Social Disorganization Theory (Shaw & McKay) and Routine Activity Theory
2. Relaxed the statistical criteria (such as communality thresholds) for theoretically important variables
3. Retained more variables per pillar to ensure comprehensive coverage of our theoretical constructs
4. Balanced statistical qualities with theoretical importance using a weighted scoring approach

This modified approach ensures that our final indicator set captures the multidimensional nature of community crime risk as conceptualized in our theoretical framework, while still maintaining statistical validity.


### Implications for Next Steps

The final set of indicators identified through multivariate analysis will be:
1. Re-normalized in Step 5
2. Weighted and aggregated in Step 6

This ensures that the composite indicators will be built on statistically sound foundations with:
- Reduced redundancy between indicators
- Balanced representation across conceptual dimensions
- Enhanced interpretability of results
- Strong theoretical grounding in criminological theory
