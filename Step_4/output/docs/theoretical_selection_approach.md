# Theoretical Approach to Variable Selection in Step 4

## Introduction

This document explains our modified approach to variable selection in Step 4, which prioritizes theoretical relevance while still considering statistical properties.

## Rationale for the Modified Approach

While traditional multivariate analysis often relies heavily on statistical criteria like communality and collinearity to select variables, we have modified our approach for the following reasons:

1. **Theoretical Framework Fidelity**: Our Community Crime-Risk Index (CCRI) is grounded in established criminological theories, particularly Social Disorganization Theory (Shaw & McKay) and Routine Activity Theory. Statistical criteria alone may eliminate variables that are conceptually central to these theories.

2. **Balanced Representation**: We need adequate representation from all four pillars (Demographics, Income, Housing, Crime) to create a theoretically sound composite index.

3. **Future Normalization and Weighting**: For Step 5 (normalization) and Step 6 (weighting), we need a more comprehensive set of variables to ensure our index captures the multidimensional nature of community crime risk.

4. **Avoiding Oversimplification**: Reducing our dataset too aggressively based solely on statistical criteria risks oversimplifying the complex phenomenon we're studying.

## Methodology Changes

We've implemented the following changes to our variable selection process:

1. **Increased Minimum Variables Per Pillar**: We've increased the minimum number of indicators per pillar from 2 to 4.

2. **Identification of Theoretically Important Variables**: We've explicitly identified variables that are theoretically important for each pillar, based on our research framework:

   - **Demographics Pillar**: 
     - `racepctblack`: Important for measuring racial composition related to social disorganization
     - `racePctHisp`: Important for ethnic heterogeneity assessment
     - `pctUrban`: Central to Shaw & McKay's work on urbanization and crime
     - `PctNotSpeakEnglWell`: Indicator of social isolation that may reduce collective efficacy

   - **Income Pillar**:
     - `PctPopUnderPov`: Fundamental measure of concentrated disadvantage
     - `medIncome`: Core socioeconomic indicator affecting guardian capability
     - `pctWPubAsst`: Marker of economic vulnerability and dependency
     - `PctUnemployed`: Key predictor of motivated offenders in routine activity theory

   - **Housing Pillar**:
     - `PctHousOccup`: Indicator of neighborhood stability and vacant property presence
     - `PctSameHouse85`: Direct measure of residential stability/turnover
     - `PctVacantBoarded`: Strong indicator of physical disorder
     - `PctHousNoPhone`: Measure of guardianship capability (ability to call for help)
     - `PctFam2Par`: Two-parent families as informal social control mechanism

   - **Crime Pillar**:
     - `ViolentCrimesPerPop`: Comprehensive violent crime measure
     - `murdPerPop`: Most serious violent crime indicator
     - `robbbPerPop`: Property crime with violence element
     - `autoTheftPerPop`: Property crime indicator (vehicle theft)
     - `arsonsPerPop`: Property destruction indicator

3. **Modified Importance Score Calculation**: We've adjusted our importance scoring algorithm to give greater weight (50%) to theoretical importance, while still considering statistical properties like communality (20%), uniqueness (15%), and PC representation (15%).

4. **Relaxed Collinearity Threshold**: We've increased the collinearity threshold from 0.9 to 0.95, meaning we only consider variables with correlations above 0.95 to be redundant. This less strict threshold allows more theoretically important variables to be retained.

5. **Retention of Correlated Variables**: We've further relaxed our stance on multicollinearity, allowing the retention of theoretically important variables even when they are highly correlated with others.

## Implications for the Analysis

This approach results in retaining more variables than a purely statistical selection would recommend. However, it ensures that our composite index:

1. Maintains strong theoretical validity
2. Captures all key dimensions of our research framework
3. Provides sufficient variables for meaningful weighting and aggregation in later steps
4. Represents each pillar adequately

While this approach may introduce some statistical redundancy, the benefits of theoretical coherence and comprehensive coverage outweigh these concerns for our specific research objectives.

## References for Theoretical Decisions

Our selection of theoretically important variables is supported by the following research:

1. Shaw, C. R., & McKay, H. D. (1942). Juvenile delinquency and urban areas. Chicago: University of Chicago Press.
2. Sampson, R. J., & Groves, W. B. (1989). Community structure and crime: Testing social-disorganization theory. American Journal of Sociology, 94(4), 774-802.
3. Cohen, L. E., & Felson, M. (1979). Social change and crime rate trends: A routine activity approach. American Sociological Review, 44, 588-608.
4. Sampson, R. J., Raudenbush, S. W., & Earls, F. (1997). Neighborhoods and violent crime: A multilevel study of collective efficacy. Science, 277(5328), 918-924.
5. OECD/EC Handbook on Constructing Composite Indicators (2008). Methodological framework for balancing statistical requirements with subject-matter knowledge.

These sources validate our decision to prioritize theoretically important variables in our final selection, even when they may not meet strict statistical thresholds. 