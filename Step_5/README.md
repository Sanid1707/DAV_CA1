# Step 5: Normalization

## Overview

This step normalizes the selected indicators from Step 4 to create comparable measures for the Community Crime-Risk Index. The normalization process ensures that all indicators are on the same scale (0-1) and have the same polarity (higher values represent better performance).

## Key Components

### Indicator Polarity Table

A polarity table was created for all 18 indicators, identifying which ones have positive polarity (↑ good) or negative polarity (↓ good). The negative polarity indicators were inverted before normalization to ensure consistent direction.

### Outlier and Skewness Analysis

Boxplots and descriptive statistics were generated to check for outliers and skewness. No additional trimming was performed before normalization, as Min-Max scaling naturally accommodates outliers.

### Normalization Method

**Min-Max normalization** was chosen for its intuitive [0, 1] scale and interpretability for policy dashboards. This method preserves the relative distances between values and creates a standardized range easily understandable by stakeholders.

### Inversion Formula

For negative polarity indicators, the following inversion formula was applied:

$$x_i^{inv} = x_{max} - x_i$$

### Min-Max Formula

After inversion, all indicators were normalized using the Min-Max formula:

$$I_i = \frac{x_i^{inv} - x_{min}^{inv}}{x_{max}^{inv} - x_{min}^{inv}}$$

### Quality Checks

Quality checks were performed to ensure:
- All normalized indicators have min ≈ 0 and max ≈ 1
- No NaN values in the normalized dataset
- Proper inversion of negative polarity indicators

### Sensitivity Analysis

As a bonus, Z-score normalization was also applied to compare with Min-Max normalization. Rank differences were calculated to assess the sensitivity of indicator values to the normalization method.

## Output Files

- `step5_normalized_dataset.csv`: The normalized dataset ready for Step 6 (weighting and aggregation)
- `indicator_descriptive_stats.csv`: Descriptive statistics of indicators before normalization
- `indicator_polarity_table.csv`: Table of indicators with units and polarity information
- `normalization_method_comparison.csv`: Comparison of Min-Max vs. Z-score normalization

## Directory Structure

- `code/`: Contains the Python script implementing the normalization process
- `docs/`: Contains documentation on the methodology and approach
- `output/`: Contains the normalized dataset and analysis results
  - `figures/`: Contains boxplots and other visualizations

## Conclusion

Step 5 successfully normalized all indicators for the Community Crime-Risk Index, preparing them for weighting and aggregation in Step 6. The Min-Max normalization approach ensures comparability while maintaining interpretability for policy applications. 