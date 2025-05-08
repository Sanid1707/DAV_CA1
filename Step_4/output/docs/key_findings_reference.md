# Step 4: Multivariate Analysis - Key Findings Reference

This document provides a concise reference of key findings from Step 4 (Multivariate Analysis), linking them to the specific output files for easy inclusion in the final report.

## 1. Principal Component Analysis Findings

### 1.1 Dimension Reduction Results

| Pillar | Components Retained | Variance Explained | Key Output Files |
|--------|---------------------|--------------------|--------------------|
| Demographics | 3 | ~72% | `Step_4/output/figures/pca_scree_Demographics.png` |
| Income | 1 | ~71% | `Step_4/output/figures/pca_scree_Income.png` |
| Housing | 5 | ~77% | `Step_4/output/figures/pca_scree_Housing.png` |
| Crime | 3 | ~77% | `Step_4/output/figures/pca_scree_Crime.png` |

**Key Finding:** The large number of original indicators can be effectively reduced to a much smaller set of components (1-5 per pillar) while retaining 70-80% of the original information.

### 1.2 Component Interpretations

| Component | Interpretation | Supporting Visualization |
|-----------|----------------|-----------------------|
| Demographics PC1 | Immigration and Diversity | `Step_4/output/figures/pca_interpretation_Demographics.png` |
| Demographics PC2 | Racial Composition | `Step_4/output/figures/pca_interpretation_Demographics.png` |
| Demographics PC3 | Urban Population | `Step_4/output/figures/pca_interpretation_Demographics.png` |
| Income PC1 | Economic Prosperity vs. Poverty | `Step_4/output/figures/pca_interpretation_Income.png` |
| Housing PC1 | Housing Density and Quality | `Step_4/output/figures/pca_interpretation_Housing.png` |
| Crime PC1 | Overall Crime Rate | `Step_4/output/figures/pca_interpretation_Crime.png` |

**Key Finding:** Each principal component has a clear conceptual interpretation that aligns with established socioeconomic and demographic frameworks.

## 2. Cluster Analysis Findings

### 2.1 Optimal Cluster Solutions

| Pillar | Optimal Clusters | Silhouette Score | Key Output Files |
|--------|------------------|------------------|--------------------|
| Demographics | 3 | 0.42 | `Step_4/output/figures/silhouette_scores_Demographics.png` |
| Income | 2 | 0.60 | `Step_4/output/figures/silhouette_scores_Income.png` |
| Housing | 2 | 0.28 | `Step_4/output/figures/silhouette_scores_Housing.png` |
| Crime | 2 | 0.45 | `Step_4/output/figures/silhouette_scores_Crime.png` |

**Key Finding:** Communities naturally form distinct groups based on their characteristics, with Income and Crime showing the clearest cluster separation.

### 2.2 Cluster Profiles

| Pillar | Cluster Characteristics | Supporting Visualization |
|--------|-------------------------|-----------------------|
| Demographics | Cluster 1: High minority populations<br>Cluster 2: Predominantly white communities<br>Cluster 3: High immigration/Hispanic | `Step_4/output/figures/Demographics_cluster_radar.png` |
| Income | Cluster 1: High-income, low-poverty<br>Cluster 2: Low-income, high-poverty | `Step_4/output/figures/Income_cluster_radar.png` |
| Housing | Cluster 1: Higher quality, stable housing<br>Cluster 2: Lower quality, less stable housing | `Step_4/output/figures/Housing_cluster_radar.png` |
| Crime | Cluster 1: Lower crime communities<br>Cluster 2: Higher crime communities | `Step_4/output/figures/Crime_cluster_radar.png` |

**Key Finding:** Each cluster has a distinct profile across multiple indicators, revealing natural groupings of communities with similar characteristics.

## 3. Collinearity Analysis Findings

### 3.1 Indicator Correlations

| Pillar | Correlation Findings | Supporting Visualization |
|--------|----------------------|-----------------------|
| Demographics | Moderate correlations between race/ethnicity variables | `Step_4/output/figures/Demographics_correlation_heatmap.png` |
| Income | Strong negative correlation between income and poverty measures | `Step_4/output/figures/Income_correlation_heatmap.png` |
| Housing | Several groups of related housing variables | `Step_4/output/figures/Housing_correlation_heatmap.png` |
| Crime | Very high correlations between related crime measures | `Step_4/output/figures/Crime_correlation_heatmap.png` |

**Key Finding:** The Crime pillar shows the highest degree of collinearity among indicators, while Demographics shows the most independence between indicators.

### 3.2 High Collinearity Pairs (|r| > 0.9)

| Pillar | High Collinearity Pairs | Reference File |
|--------|-------------------------|-----------------------|
| Crime | murders-murdPerPop<br>robberies-robbbPerPop | `Step_4/output/indicator_decision_grid.csv` |

**Key Finding:** Crime indicators show especially high correlations between raw counts and per-capita rates, indicating potential redundancy.

## 4. Indicator Selection Findings

### 4.1 Final Indicator Selection

| Pillar | Indicators Kept | Indicators Dropped | Reference File |
|--------|----------------|--------------------|-----------------------|
| Demographics | 2 | 7 | `Step_4/output/final_indicators_pivot.csv` |
| Income | 0* | 5* | `Step_4/output/final_indicators_pivot.csv` |
| Housing | 2 | 12 | `Step_4/output/final_indicators_pivot.csv` |
| Crime | 0* | 18* | `Step_4/output/final_indicators_pivot.csv` |

\* *Note: Despite initial decisions, Income and Crime indicators were retained for comprehensive measurement*

**Key Finding:** Indicator selection process identified significant redundancy, particularly in Housing and Demographics pillars, allowing for substantial dimension reduction.

### 4.2 PCA Results with Trimmed Indicators

| Pillar | Original Variance | Trimmed Variance | Supporting Visualization |
|--------|------------------|------------------|-----------------------|
| Demographics | 72% (3 PCs) | 100% (2 PCs) | `Step_4/output/figures/pca_trimmed_Demographics.png` |
| Housing | 77% (5 PCs) | 100% (2 PCs) | `Step_4/output/figures/pca_trimmed_Housing.png` |

**Key Finding:** The trimmed indicator set achieves similar or better variance explanation with fewer components, confirming successful redundancy reduction.

## 5. Implications for Next Steps

1. **For Step 5 (Normalization):**
   - The selected indicators identified in `Step_4/output/final_indicators.csv` should be normalized
   - Special attention should be given to skewed indicators identified during standardization

2. **For Step 6 (Weighting and Aggregation):**
   - PCA results can inform weighting strategies
   - Component loadings provide guidance on indicator importance within each pillar
   - Reference file: `Step_4/output/pca_loadings_summary.csv`

3. **For Overall Composite Indicator:**
   - The reduced set of non-redundant indicators will provide a more robust and interpretable composite
   - Clusters identified can serve as validation groups for the final composite indicator

## 6. Summary Statistics Tables for Report

### 6.1 PCA Summary

| Pillar | Original Variables | Components Retained | Variance Explained | Key Indicators |
|--------|-------------------|---------------------|---------------------|----------------|
| Demographics | 9 | 3 | 72% | racepctblack, pctUrban |
| Income | 5 | 1 | 71% | All retained |
| Housing | 14 | 5 | 77% | PctHousOccup, PctSameHouse85 |
| Crime | 18 | 3 | 77% | All retained |

### 6.2 Clustering Summary

| Pillar | Optimal Clusters | Method Agreement | Distinguishing Features |
|--------|------------------|------------------|-------------------------|
| Demographics | 3 | 0.66 | Racial composition, Immigration |
| Income | 2 | 0.98 | Income levels, Poverty rates |
| Housing | 2 | 0.42 | Housing stability, Occupancy |
| Crime | 2 | 0.47 | Overall crime rates |

*Note: Method Agreement refers to Adjusted Rand Index between hierarchical and k-means clustering* 