
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
- Final indicators: 2
- Variance explained by PC1: 0.51 (50.8%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: racepctblack, pctUrban

#### Income Pillar

- Original indicators: 5
- Final indicators: 2
- Variance explained by PC1: 0.89 (89.4%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: pctWPubAsst, PctPopUnderPov

#### Housing Pillar

- Original indicators: 14
- Final indicators: 2
- Variance explained by PC1: 0.60 (60.4%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: PctHousOccup, PctSameHouse85

#### Crime Pillar

- Original indicators: 18
- Final indicators: 2
- Variance explained by PC1: 0.89 (88.6%)
- Total variance explained: 1.00 (100.0%)
- Final indicators: arsons, arsonsPerPop


### Cluster Analysis

Cluster analysis was performed using both hierarchical and k-means clustering methods on the principal component scores. The optimal number of clusters was determined using silhouette scores.

Detailed cluster analysis information not available.


### Indicator Selection

The following summarizes the final indicator selection based on multivariate analysis:

#### Demographics Pillar

- Selected indicators: 2
- Indicators: racepctblack, pctUrban
- Dropped indicators: population, racePctWhite, racePctAsian, racePctHisp, PctImmigRec5, PctRecentImmig, PctNotSpeakEnglWell

#### Income Pillar

- Selected indicators: 2
- Indicators: pctWPubAsst, PctPopUnderPov
- Dropped indicators: medIncome, pctWSocSec, PctUnemployed

#### Housing Pillar

- Selected indicators: 2
- Indicators: PctHousOccup, PctSameHouse85
- Dropped indicators: PersPerFam, PctFam2Par, PctLargHouseFam, PersPerRentOccHous, PctPersDenseHous, PctVacantBoarded, PctHousNoPhone, PctWOFullPlumb, OwnOccQrange, RentQrange, NumInShelters, PctUsePubTrans

#### Crime Pillar

- Selected indicators: 2
- Indicators: arsons, arsonsPerPop
- Dropped indicators: murders, murdPerPop, rapes, rapesPerPop, robberies, robbbPerPop, assaults, assaultPerPop, burglaries, burglPerPop, larcenies, larcPerPop, autoTheft, autoTheftPerPop, ViolentCrimesPerPop, nonViolPerPop


### Implications for Next Steps

The final set of indicators identified through multivariate analysis will be:
1. Re-normalized in Step 5
2. Weighted and aggregated in Step 6

This ensures that the composite indicators will be built on statistically sound foundations with:
- Reduced redundancy between indicators
- Balanced representation across conceptual dimensions
- Enhanced interpretability of results
