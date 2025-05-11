# Community Crime Vulnerability and Exposure Index (CCVEI)

## Project Overview

This repository contains the implementation of a Community Crime Vulnerability and Exposure Index (CCVEI), a composite indicator measuring both community-level vulnerability factors and crime exposure. The index is built on a dual theoretical framework that integrates Social Vulnerability Theory, Economic Strain Theory, Social Disorganization Theory, and Crime Pattern Theory.

## Theoretical Framework

The CCVEI is constructed around four theoretical pillars, divided into vulnerability and exposure components:

### Vulnerability Factors:
- **Demographics**: Demographic composition and heterogeneity factors that influence community social cohesion and vulnerability
- **Income**: Economic factors related to poverty, unemployment, and financial strain that affect community resilience
- **Housing**: Physical environment, residential stability, and housing conditions that impact community vulnerability

### Exposure Factor:
- **Crime**: Direct crime measures across different crime categories representing current exposure to criminal activity

## Project Structure

The project follows the OECD/JRC Handbook on Constructing Composite Indicators methodology and is organized into steps:

- **Step 1**: Theoretical Framework
- **Step 2**: Data Collection & Initial Processing
- **Step 3**: Multivariate Analysis & Indicator Selection
- **Step 4**: Imputation of Missing Data
- **Step 5**: Normalization of Indicators
- **Step 6**: Weighting & Aggregation
- **Step 8**: Decomposition & Profiling
- **Step 9**: External Linkage & Criterion Validity

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
- Composite CCVEI scores under three weighting scenarios
- Community rankings
- Visualizations of results
- Thorough documentation of methodology

## How to Run

Navigate to each step's directory and run the corresponding Python script:

```bash
# For Step 6:
python Step_6/code/weighting_aggregation.py

# For the final analysis (Step 8 & 9):
python run_final_analysis.py
```

## Documentation

Each step includes detailed documentation explaining the methodology, decisions, and results:

- `Step_1/Theoretical_Framework.md`: Detailed explanation of the CCVEI framework and its theoretical foundations
- `Step_6/docs/weighting_aggregation_methodology.md`: In-depth explanation of the weighting and aggregation approach
- `Step_6/docs/table_6.1_normalized_indicators_by_pillar.md`: Overview of indicators by pillar
- `Step_9/docs/validation_report.md`: Validation analysis of the CCVEI

## Theoretical Implications

The CCVEI provides insights into the multidimensional nature of community crime vulnerability and exposure, highlighting:

1. **Dual-Component Framework**: How vulnerability factors interact with current crime exposure
2. **Compensatory Effects**: How strengths in some areas can offset weaknesses in others
3. **Stakeholder Perspectives**: How different weighting scenarios produce varying community assessments

## Future Directions

Future improvements to the CCVEI could include:

- Incorporating longitudinal data to track changes over time
- Adding qualitative indicators of community cohesion
- Developing web-based visualization tools for community stakeholders
- Validating index results against independent measures of community well-being

## References

- OECD/JRC. (2008). Handbook on Constructing Composite Indicators: Methodology and User Guide. OECD Publishing.
- Cutter, S. L., Boruff, B. J., & Shirley, W. L. (2003). Social vulnerability to environmental hazards. Social Science Quarterly, 84(2), 242-261.
- Shaw, C. R., & McKay, H. D. (1942). Juvenile delinquency and urban areas. University of Chicago Press.
- Brantingham, P. J., & Brantingham, P. L. (1993). Environment, routine and situation: Toward a pattern theory of crime. Advances in Criminological Theory, 5(2), 259-294.
- Sampson, R. J., Raudenbush, S. W., & Earls, F. (1997). Neighborhoods and violent crime: A multilevel study of collective efficacy. Science, 277(5328), 918-924.
- Merton, R. K. (1938). Social structure and anomie. American Sociological Review, 3(5), 672-682. 