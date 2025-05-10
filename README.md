# Community Crime-Risk Index (CRI)

## Project Overview

This repository contains the implementation of a Community Crime-Risk Index (CRI), a composite indicator measuring community-level risk factors associated with crime. The index is built on a theoretical framework that integrates Social Disorganization Theory, Routine Activity Theory, and other criminological perspectives.

## Theoretical Framework

The CRI is constructed around four theoretical pillars:

- **Demographics**: Demographic composition and heterogeneity factors that influence community social cohesion
- **Income**: Economic factors related to poverty, unemployment, and financial strain
- **Housing**: Physical environment, residential stability, and family structure
- **Crime**: Direct crime measures across different crime categories

## Project Structure

The project follows the OECD/JRC Handbook on Constructing Composite Indicators methodology and is organized into steps:

- **Step 2**: Data Collection & Initial Processing
- **Step 3**: Multivariate Analysis & Indicator Selection
- **Step 4**: Imputation of Missing Data
- **Step 5**: Normalization of Indicators
- **Step 6**: Weighting & Aggregation (Current Step)

## Step 6: Weighting & Aggregation

Step 6 applies weighting and aggregation methods to transform normalized indicators into a composite index:

### Key Features

1. **Within-Pillar Weighting**:
   - Demographics: PCA-variance weights
   - Income: PCA-variance weights
   - Housing: PCA-variance weights
   - Crime: Equal weights (EW)

2. **Double-Counting Check**:
   - Correlation analysis to identify highly correlated indicators (|r| > 0.9)
   - Weight adjustment to prevent double-counting

3. **Aggregation Method**:
   - Linear weighted mean for combining indicators into pillars
   - Allows for partial compensability between indicators

4. **Cross-Pillar Weighting**:
   - Scenario 1: Equal weights (0.25 for each pillar)
   - Scenario 2: PCA-variance weights
   - Scenario 3: Stakeholder budget allocation weights

5. **Sensitivity Analysis**:
   - Rank shifts between different weighting scenarios
   - Visual comparison of community rankings

### Outputs

- Pillar scores for each community
- Composite CRI scores under three weighting scenarios
- Community rankings
- Visualizations of results
- Thorough documentation of methodology

## How to Run

Navigate to each step's directory and run the corresponding Python script:

```bash
# For Step 6:
python Step_6/code/weighting_aggregation.py
```

## Documentation

Each step includes detailed documentation explaining the methodology, decisions, and results:

- `Step_6/docs/weighting_aggregation_methodology.md`: In-depth explanation of the weighting and aggregation approach
- `Step_6/docs/table_6.1_normalized_indicators_by_pillar.md`: Overview of indicators by pillar
- `Step_6/docs/compensability_discussion.md`: Discussion of compensability in the aggregation method

## Theoretical Implications

The CRI provides insights into the multidimensional nature of community crime risk, highlighting:

1. **Risk Factor Interactions**: How different risk factors interact across domains
2. **Compensatory Effects**: How strengths in some areas can offset weaknesses in others
3. **Stakeholder Perspectives**: How different weighting scenarios produce varying community assessments

## Future Directions

Future improvements to the CRI could include:

- Incorporating longitudinal data to track changes over time
- Adding qualitative indicators of community cohesion
- Developing web-based visualization tools for community stakeholders
- Validating index results against independent measures of community well-being

## References

- OECD/JRC. (2008). Handbook on Constructing Composite Indicators: Methodology and User Guide. OECD Publishing.
- Shaw, C. R., & McKay, H. D. (1942). Juvenile delinquency and urban areas. University of Chicago Press.
- Cohen, L. E., & Felson, M. (1979). Social change and crime rate trends: A routine activity approach. American Sociological Review, 44(4), 588-608.
- Sampson, R. J., Raudenbush, S. W., & Earls, F. (1997). Neighborhoods and violent crime: A multilevel study of collective efficacy. Science, 277(5328), 918-924. 