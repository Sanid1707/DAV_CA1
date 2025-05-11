# Community Risk Index: Step 6 – Weighting & Aggregation

## 1. Theoretical Framework

The Community Risk Index is built on four theoretical pillars that together provide a holistic view of crime risk in communities:

- **Demographics**: Captures the demographic composition and heterogeneity of communities, reflecting Social Disorganization Theory's emphasis on how demographic factors affect informal social control.
- **Income**: Measures economic deprivation and financial strain, which create strain and reduce resources for community guardianship.
- **Housing**: Represents the physical environment, residential stability, and family structure that shape both opportunities for crime and community cohesion.
- **Crime**: Consists of various crime measures that serve as both outcome indicators and proxies for overall community safety.

Each of these pillars represents a critical dimension in criminological theory, particularly Social Disorganization Theory (Shaw & McKay) and Routine Activity Theory (Cohen & Felson), which link community characteristics to crime rates.

## 2. Inventory of Normalized Indicators by Pillar

After normalization in Step 5, the following indicators feed into each theoretical pillar:

| Pillar | Indicators | Theoretical Relevance |
|--------|------------|------------------------|
| **Demographics** | racepctblack, racePctHisp, pctUrban, PctNotSpeakEnglWell | Measures demographic heterogeneity and characteristics that can influence social cohesion |
| **Income** | medIncome, pctWPubAsst, PctPopUnderPov, PctUnemployed | Captures economic deprivation that can lead to strain and reduced guardianship |
| **Housing** | PctFam2Par, PctHousOccup, PctVacantBoarded, PctHousNoPhone, PctSameHouse85 | Reflects physical environment, stability, and supervision capacity |
| **Crime** | murdPerPop, robbbPerPop, autoTheftPerPop, arsonsPerPop, ViolentCrimesPerPop | Direct measures of various crime types that together provide a comprehensive picture of community safety |

All indicators have been normalized to a 0-1 scale with consistent polarity (higher values represent better outcomes, i.e., lower crime risk).

## 3. Weighting Logic Within Each Pillar

Different weighting approaches were chosen for each pillar based on their theoretical characteristics and statistical properties:

### Demographics Pillar: PCA-Variance Weights
- **Justification**: The demographics indicators capture related but distinct aspects of community composition. PCA weighting helps to account for the shared variance among these variables while preserving their unique contributions. This aligns with theoretical understanding that different demographic factors may have overlapping effects on community risk.
- **Implementation**: Principal Component Analysis was applied to the indicators, and weights were derived from the explained variance of each component, ensuring that indicators with higher explanatory power receive proportionally higher weights.

### Income Pillar: PCA-Variance Weights
- **Justification**: Income-related indicators are closely related conceptually but measure different aspects of economic disadvantage. PCA weights account for the correlation structure while maintaining the distinct contribution of each measure (e.g., poverty rate vs. unemployment).
- **Implementation**: PCA-based weights ensure that the pillar score is not unduly influenced by any single economic measure while still capturing the overall concept of economic disadvantage.

### Housing Pillar: PCA-Variance Weights
- **Justification**: Housing indicators cover diverse aspects from physical conditions to stability. PCA weights help balance these different dimensions based on their empirical relationships in the data.
- **Implementation**: The variance-based weighting adjusts for the fact that some housing indicators may be more relevant in certain contexts while maintaining theoretical coherence.

### Crime Pillar: Equal Weights (EW)
- **Justification**: All crime types are considered equally important for assessing overall community safety. No single crime type should dominate the index, as communities may have different crime patterns but similar overall risk levels.
- **Implementation**: A simple arithmetic mean gives equal importance to each crime indicator, ensuring balanced representation across crime categories.

## 4. Double-Counting Check and Weight Adjustment

Within each pillar, a correlation analysis was performed to identify potential redundancy (double-counting) among indicators:

1. **Correlation matrices** were computed for each pillar and visualized as heatmaps.
2. **Highly correlated pairs** (|r| > 0.9) were flagged for potential weight adjustment.
3. **Weight adjustment approach**: When high correlation was detected, weights for both affected indicators were reduced by 50% and then renormalized to sum to 1. This adjustment prevents overemphasis on essentially the same information.

This approach follows the OECD Handbook's recommendation to address collinearity when building composite indicators.

## 5. Aggregation Method: Linear Mean

For aggregating indicators into pillar scores, we used a linear weighted mean:

$P_j = \sum_{i=1}^{n} w_i \times I_i$

Where:
- $P_j$ is the score for pillar $j$
- $w_i$ is the weight for indicator $i$
- $I_i$ is the normalized value of indicator $i$

This linear aggregation method allows for **partial compensability** between indicators, meaning that a deficit in one indicator can be partially offset by a surplus in another. This aligns with the theoretical understanding that community risk factors interact with each other and can be mitigated by strengths in other areas.

## 6. Cross-Pillar Weighting Scheme

Three scenarios were developed to explore different approaches to weighting the four pillars:

### Scenario 1: Equal Weights (EW)
- **Justification**: All four risk domains (Demographics, Income, Housing, Crime) are considered equally relevant to overall community risk, with no theoretical basis to prioritize one domain over another.
- **Implementation**: Each pillar receives a weight of 0.25, reflecting equal importance.

### Scenario 2: PCA-Variance Weights
- **Justification**: Allows the data structure to determine the relative importance of each pillar based on its statistical properties.
- **Implementation**: Weights derived from principal component analysis of the pillar scores, with weights proportional to the explained variance.

### Scenario 3: Stakeholder Budget Allocation
- **Justification**: Reflects the priorities of different stakeholders in addressing community crime risk.
- **Implementation**: Hypothetical allocation based on stakeholder priorities:
  - Law Enforcement (Crime): 30%
  - Local Government (Income): 30%
  - Housing Authorities (Housing): 20%
  - Community Organizations (Demographics): 20%

This approach provides flexibility in considering different perspectives on community risk factors.

## 7. Composite Risk Index Calculation

The final Community Risk Index (CRI) is calculated as a weighted sum of the pillar scores:

$CRI = \sum_{j=1}^{4} w_j \times P_j$

Where:
- $w_j$ is the weight for pillar $j$
- $P_j$ is the score for pillar $j$

This calculation is performed for each of the three weighting scenarios, producing three versions of the CRI. 

## 8. Compensability and Theoretical Implications

The linear aggregation method used in this index allows for partial compensability between indicators and pillars. This means:

- **Within pillars**: Strengths in one indicator (e.g., low vacancy rates) can partially offset weaknesses in another (e.g., low homeownership).
- **Between pillars**: Strong performance in one domain (e.g., Income) can partially compensate for weaknesses in another (e.g., Demographics).

This compensability aligns with criminological theory, which suggests that protective factors in one domain can mitigate risk factors in another. For example, strong family structures (Housing pillar) can help reduce the negative effects of economic disadvantage (Income pillar).

However, the use of multiple scenarios acknowledges that different stakeholders may have different views on the relative importance of various risk factors.

## 9. Sensitivity Analysis and Robustness

To assess the robustness of the Community Risk Index, a sensitivity analysis was conducted:

1. **Rank shifts**: Communities were ranked under each weighting scenario, and rank differences were calculated.
2. **Visualization**: Communities with the largest rank shifts were visualized to identify which are most sensitive to methodological choices.
3. **Interpretation**: Large rank shifts indicate communities where the risk profile is more dependent on which factors are emphasized.

This analysis helps identify communities where targeted interventions might be most effective depending on stakeholder priorities.

## 10. Quality Assurance

Several quality checks were performed to ensure the methodological soundness of the index:

1. **Range verification**: Confirmed that all pillar scores and composite scores span the 0-1 range.
2. **Missing value check**: Verified that no missing values were introduced during the weighting and aggregation process.
3. **Weight normalization**: Ensured that weights within each pillar and across pillars sum to 1.
4. **Documentation**: All methodological choices and their justifications have been thoroughly documented.

These quality assurance steps ensure that the Community Risk Index is technically sound and transparent in its construction.

## 11. Conclusion

The weighting and aggregation approach for the Community Risk Index balances theoretical considerations with empirical relationships in the data. By using multiple weighting scenarios, the index acknowledges different perspectives on community risk while maintaining methodological rigor. The resulting composite scores provide a comprehensive assessment of community crime risk that can inform policy decisions and resource allocation. 