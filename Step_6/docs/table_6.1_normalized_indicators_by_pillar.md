# Table 6.1: Normalized Indicators by Pillar

This table provides an overview of all normalized indicators organized by theoretical pillars, including their descriptions, theoretical relevance, and the weighting approach applied within each pillar.

## Demographics Pillar

**Weighting Approach**: PCA-Variance Weights

| Indicator | Description | Theoretical Relevance |
|-----------|-------------|------------------------|
| racepctblack | Percentage of population that is Black | Demographic heterogeneity affecting social cohesion (Shaw & McKay) |
| racePctHisp | Percentage of population that is Hispanic/Latino | Demographic heterogeneity affecting social cohesion (Shaw & McKay) |
| pctUrban | Percentage of population living in urban areas | Urban density associated with increased crime opportunities (Cohen & Felson) |
| PctNotSpeakEnglWell | Percentage of population not speaking English well | Language barriers that can reduce community integration |

## Income Pillar

**Weighting Approach**: PCA-Variance Weights

| Indicator | Description | Theoretical Relevance |
|-----------|-------------|------------------------|
| medIncome | Median household income | Economic resources available for guardianship (Cohen & Felson) |
| pctWPubAsst | Percentage with public assistance | Economic disadvantage indicator (Shaw & McKay) |
| PctPopUnderPov | Percentage of population under poverty line | Concentrated disadvantage (Sampson) |
| PctUnemployed | Unemployment rate | Strain theory; lack of legitimate opportunities (Merton) |

## Housing Pillar

**Weighting Approach**: PCA-Variance Weights

| Indicator | Description | Theoretical Relevance |
|-----------|-------------|------------------------|
| PctFam2Par | Percentage of families with two parents | Family structure affecting supervision (Sampson) |
| PctHousOccup | Percentage of houses occupied | Occupancy as deterrent to crime (Newman) |
| PctVacantBoarded | Percentage of houses vacant/boarded | Physical decay indicator; broken windows theory (Wilson & Kelling) |
| PctHousNoPhone | Percentage of houses without phone | Material deprivation; reduced ability to contact authorities |
| PctSameHouse85 | Percentage in same house since 1985 | Residential stability affecting social ties (Sampson) |

## Crime Pillar

**Weighting Approach**: Equal Weights (EW)

| Indicator | Description | Theoretical Relevance |
|-----------|-------------|------------------------|
| murdPerPop | Murders per 100K population | Violent crime indicator with high reliability |
| robbbPerPop | Robberies per 100K population | Property crime with personal confrontation |
| autoTheftPerPop | Auto thefts per 100K population | Property crime with good reporting rates |
| arsonsPerPop | Arsons per 100K population | Deliberate property destruction indicator |
| ViolentCrimesPerPop | Violent crimes per 100K population | General measure of community violence |

*Note: All indicators have been normalized to a 0-1 scale where higher values indicate better outcomes (lower risk).*

## Weighting Justification

### Demographics Pillar (PCA Weights)
The demographic variables represent distinct but related aspects of community composition. PCA weighting helps identify the underlying factors driving demographic variation while accounting for any overlap between indicators.

### Income Pillar (PCA Weights)
Economic indicators often exhibit correlation but measure distinct aspects of disadvantage. PCA weights ensure that the most informative aspects of economic conditions receive appropriate emphasis while controlling for shared variance.

### Housing Pillar (PCA Weights)
Housing indicators span physical conditions, stability, and family structure. PCA helps determine how these different dimensions contribute to the overall housing environment while accounting for their interrelationships.

### Crime Pillar (Equal Weights)
Different crime types are considered equally important for community safety assessment. Equal weighting ensures that no single crime type dominates the index and that communities with different crime patterns but similar overall risk levels are treated comparably. 