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

2. **Identification of Theoretically Important Variables**: We've explicitly identified variables that are theoretically important for each pillar, based on our research framework.

3. **Modified Importance Score Calculation**: We've adjusted our importance scoring algorithm to give greater weight (50%) to theoretical importance, while still considering statistical properties like communality (20%), uniqueness (15%), and PC representation (15%).

4. **Retention of Correlated Variables**: We've relaxed our stance on multicollinearity, allowing the retention of theoretically important variables even when they are highly correlated with others.

## Implications for the Analysis

This approach results in retaining more variables than a purely statistical selection would recommend. However, it ensures that our composite index:

1. Maintains strong theoretical validity
2. Captures all key dimensions of our research framework
3. Provides sufficient variables for meaningful weighting and aggregation in later steps
4. Represents each pillar adequately

While this approach may introduce some statistical redundancy, the benefits of theoretical coherence and comprehensive coverage outweigh these concerns for our specific research objectives.
