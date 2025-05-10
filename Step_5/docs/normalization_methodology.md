# Step 5: Normalization Methodology

## Overview

This document describes the normalization methodology applied to the indicators selected in Step 4.

## Approach

Min-Max normalization was chosen for its intuitive [0, 1] scale and interpretability for policy dashboards. This method preserves the relative distances between values and creates a standardized range that is easily understandable by stakeholders.

## Polarity Inversion

Indicators with negative polarity (where lower values are better) were inverted before normalization using the formula:

$x_i^{inv} = x_{max} - x_i$

This ensures all indicators have the same direction (higher values are better) after normalization.

## Min-Max Normalization

After polarity inversion, all indicators were normalized using the Min-Max formula:

$I_i = \frac{x_i^{inv} - x_{min}^{inv}}{x_{max}^{inv} - x_{min}^{inv}}$

This transforms all indicators to the [0, 1] range while preserving their distribution.

## Quality Checks

The following quality checks were performed on the normalized data:

1. Verification that all normalized indicators have minimum ≈ 0 and maximum ≈ 1
2. Check for any NaN values in the normalized dataset
3. Visual inspection of distributions through boxplots

## Outlier Handling

Based on the boxplot analysis, no additional trimming was deemed necessary before normalization. The Min-Max approach naturally accommodates outliers by stretching the distribution to the [0, 1] range.

## Skewness

While some indicators showed skewness, no log transformations were applied. The skewness was within acceptable range for Min-Max normalization.

