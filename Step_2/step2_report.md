# Data Selection Process (Step 2)

## Overview
This document details the data selection process for the crime risk analysis project, following a theory-driven approach based on social disorganization theory. The process ensured that all crime indicators were preserved while maintaining data quality and addressing redundancy issues.

## Step 2.0: Conceptual Mapping
Starting with all 147 variables in the dataset, each variable was mapped to one of four theoretical pillars:
- Socio-economic Disadvantage
- Residential Instability
- Population & Demographic factors
- Crime Indicators (outputs)

Each variable was assigned a role (Input, Process, or Output) based on its theoretical function. This mapping is available in `step2_variable_mapping.csv`.

## Step 2.1: Initial Shortlist
Based on the theoretical mapping, 55 variables were shortlisted for further analysis. The distribution across pillars was:
                     Pillar  Count
           Crime Indicators     18
                      Other      1
   Population & Demographic     10
    Residential Instability      9
Socio-economic Disadvantage     17

## Step 2.2: Missing Data Analysis
Missing data was analyzed for all shortlisted variables. The top 15 variables with missing data are visualized in `step2_missing_top15.png`. Instead of treating zeros as missing in crime variables, the analysis preserved legitimate zero values, which represent valid "no crime" observations.

## Step 2.3: Variable Type Classification
Variables were classified as either Hard (quantitative) or Soft (qualitative), and as Direct measures or Proxy measures. The distributions are visualized in `step2_var_type.png` and `step2_measure_type.png`.

## Step 2.4: Pillar Distribution
The distribution of variables across the theoretical pillars is shown in `step2_vars_by_pillar.png`, confirming good representation of all aspects of the theoretical framework.

## Step 2.5: Handling Missing Data
Different thresholds were applied for different variable types:
- 40% missing threshold for non-crime variables
- 60% missing threshold for crime variables

This approach ensured that important crime indicators were not unnecessarily excluded. Variables dropped due to high missingness are listed in `step2_dropped_missing.csv`.

## Step 2.6: Constant and Duplicate Detection
No constant columns or exact duplicates were found in the dataset. If any had been found, they would have been documented in `step2_dropped_constants.csv`.

## Step 2.7: Outlier Screening
Outliers were examined for key variables from each pillar, as shown in `step2_boxplot_panel.png`. This helped identify extreme values that might influence subsequent analysis.

## Step 2.8: Correlation Analysis Within Pillars
Correlation analysis was performed within each pillar separately, rather than across the entire dataset. This approach preserved the theoretical structure while still addressing multicollinearity. The network of highly correlated variables is visualized in `step2_correlation_network.png`.

## Step 2.9: Final Correlation Analysis
A correlation heatmap of the final selected variables is provided in `step2_corr_heatmap.png`, showing the relationships between variables after all cleaning steps.

## Step 2.10: Proxy Validation
Proxy variables for socio-economic disadvantage were validated against crime outcome measures. The correlation between proxy variables and the benchmark crime indicator is visualized in `step2_proxy_cors.png`.

## Step 2.11: Final Variable Selection
The final selection includes 47 variables that:
- Match the theoretical framework
- Have acceptable levels of missing data
- Are not constant or duplicate
- Are not excessively collinear within their pillar

The final variable list is available in `step2_selected_vars.csv`.

## Step 2.12: Summary by Pillar
The final dataset includes variables from all four theoretical pillars:
- 10 Socio-economic Disadvantage variables
- 8 Residential Instability variables
- 10 Population & Demographic variables
- 18 Crime Indicators

Detailed statistics by pillar are shown in `step2_pillar_summary.png` and available in `step2_pillar_summary.csv`.

## Next Steps
With the data now properly selected and cleaned, the next step is to:
1. Consider creating a composite Crime-Risk score from the crime indicators
2. Perform feature engineering and transformation as needed
3. Proceed with exploratory analysis of relationships between input variables and crime outcomes
