# Step 4: Multivariate Analysis - Comprehensive Guide

## Introduction

Multivariate analysis is a critical step in developing robust composite indicators. It examines the underlying structure of the dataset, helps reduce dimensionality while preserving information, and identifies patterns that inform indicator selection. This document provides a comprehensive overview of the entire Step 4 process, focusing on our statistical methodology, decision-making rationale, and improvements to the standard approach.

## Overview of Step 4 Process

Step 4 consists of four main modules, each implementing key components of the multivariate analysis:

1. **Step_4.py**: Implements core PCA analysis and clustering
2. **Step_4_cluster_profiling.py**: Characterizes identified clusters
3. **Step_4_collinearity.py**: Analyzes variable correlations and redundancy
4. **Step_4_indicator_selection.py**: Makes data-driven indicator selection decisions

These modules are orchestrated by `run_step4.py`, which runs them in sequence and tracks execution status.

## Methodology and Implementations

### 4-A: Data Standardization

Before conducting multivariate analyses, variables must be in comparable units. We implemented z-standardization with special handling for skewed variables:

- Standard z-score for normally distributed variables: `(x - mean)/std`
- Log-transformation followed by z-standardization for skewed variables (skewness > 1)

This approach follows the OECD Handbook's recommendation to use appropriate transformations for skewed data (§4.1) while maintaining comparability between variables.

### 4-B & 4-C: Principal Component Analysis (PCA)

PCA reduces dimensionality by identifying orthogonal components that capture maximum variance in the data. Our implementation:

1. Performs PCA separately for each pillar to preserve domain-specific structure
2. Determines optimal number of components based on multiple criteria:
   - Kaiser criterion (eigenvalue > 1)
   - Cumulative variance threshold (≥ 60%)
   - Scree plot "elbow" analysis
   - Conceptual interpretability of components

For each pillar, we retained:
- Demographics: 3 components (explaining ~72% of variance)
- Income: 1 component (explaining ~71% of variance)
- Housing: 5 components (explaining ~77% of variance)
- Crime: 3 components (explaining ~77% of variance)

Each component has a clear interpretation based on variable loadings, providing valuable information about the underlying data structure.

### 4-D: Cluster Analysis on Communities

Cluster analysis groups similar communities based on their characteristics. We applied:

1. Hierarchical clustering (Ward's method)
2. K-means clustering
3. Silhouette analysis to determine optimal number of clusters
4. Adjusted Rand Index to measure agreement between methods
5. Stability analysis through repeated runs with different initializations

Results showed:
- Demographics: 3 distinct clusters with good separation
- Income: 2 clusters with excellent separation
- Housing: 2 clusters with moderate overlap
- Crime: 2 clusters with good separation

### 4-E: Cluster Profiling

We characterized each cluster using:
- Box plots showing distribution of original indicators within clusters
- Radar charts displaying characteristic profiles
- Heatmaps showing mean values of indicators in each cluster

This profiling revealed distinctive patterns for each cluster, providing insights into community types.

### 4-F: Collinearity Analysis

Identifying redundant information is crucial for efficient indicator selection. Our approach included:

1. Correlation analysis within each pillar
2. Hierarchical clustering of indicators based on correlation distance
3. Identification of highly correlated pairs (|r| > 0.9)
4. Visualization through dendrograms and heatmaps

Results showed:
- Demographics: Moderate correlations, no extremely high correlations
- Income: Strong negative correlation between income and poverty measures
- Housing: Several groups of related indicators
- Crime: Multiple high correlation pairs, especially between raw and per-capita metrics

### 4-G & 4-H: Indicator Selection and Validation

#### Traditional Approach Limitations

The traditional approach to indicator selection often relies solely on statistical criteria:
- Dropping variables with low communality
- Eliminating one variable from highly correlated pairs
- Using only PCA loadings for selection

This can lead to:
- Entire pillars being under-represented or eliminated
- Loss of conceptually important variables
- Excessive data reduction that compromises the composite indicator's validity

#### Our Balanced Selection Approach

We developed an enhanced, balanced indicator selection method that combines statistical quality with conceptual importance:

1. **Importance Score Calculation**:
   - Statistical quality (communality): 30% weight
   - Uniqueness (inverse of being in correlation groups): 20% weight
   - Principal component representation: 20% weight
   - Conceptual importance: 30% weight

2. **Minimum Representation Guarantee**:
   - Ensuring at least 2 indicators per pillar
   - Prioritizing indicators with high importance scores
   - Special handling for Crime and Income pillars to ensure PC1 representation

3. **Selection Process**:
   - Start with statistically recommended indicators ("Keep" decisions)
   - Add indicators from "Consider" group based on importance score when needed
   - If necessary, include highest-scoring indicators from "Drop" category to ensure pillar representation

4. **Validation Through Re-running PCA**:
   - Confirm that the selected indicators preserve explanatory power
   - Verify that principal components remain interpretable
   - Compare variance explained before and after selection

This approach draws from best practices in composite indicator methodology, particularly:

- OECD Handbook on Constructing Composite Indicators (2008), which emphasizes balancing statistical considerations with conceptual rationale
- Greco et al. (2019), "On the Methodological Framework of Composite Indices: A Review of the Issues of Weighting, Aggregation, and Robustness"
- Becker et al. (2017), "Weights and Importance in Composite Indicators: Closing the Gap"

### 4-I: Documentation and Preparation for Step 5

The final stage creates comprehensive documentation, including:
- Analysis summary
- Visualization guide
- Key findings reference
- Executive summary
- Final dataset for Step 5

## Results and Findings

### PCA Results

| Pillar | Original Variables | Components Retained | Variance Explained | Key Insights |
|--------|-------------------|---------------------|---------------------|----------------|
| Demographics | 9 | 3 | 72% | PC1: Immigration/Diversity, PC2: Racial Composition, PC3: Urban Population |
| Income | 5 | 1 | 71% | PC1: Economic Prosperity vs. Poverty |
| Housing | 14 | 5 | 77% | PC1: Housing Density, PC2-5: Various housing aspects |
| Crime | 18 | 3 | 77% | PC1: Overall Crime Rate, PC2-3: Crime type patterns |

### Cluster Analysis Results

| Pillar | Optimal Clusters | Method Agreement | Distinguishing Features |
|--------|------------------|------------------|-------------------------|
| Demographics | 3 | 0.66 | Racial composition, Immigration patterns |
| Income | 2 | 0.98 | High vs. low economic prosperity |
| Housing | 2 | 0.42 | Housing quality and stability |
| Crime | 2 | 0.47 | High vs. low crime rates |

### Indicator Selection Results

Using our balanced selection approach:

1. **Demographics Pillar**:
   - Selected key indicators representing different demographic dimensions
   - Maintained representation of major racial/ethnic groups and urbanization
   - Achieved 100% variance explanation with fewer indicators

2. **Income Pillar**:
   - Retained critical indicators of both income and poverty
   - Preserved the core economic prosperity dimension
   - Maintained excellent explanatory power

3. **Housing Pillar**:
   - Selected representative indicators from different housing aspects
   - Reduced redundancy while maintaining key information
   - Streamlined the most complex pillar

4. **Crime Pillar**:
   - Addressed high multicollinearity between crime measures
   - Selected representative indicators that capture different crime types
   - Maintained overall crime rate representation

## Discussion and Implications

### Strengths of Our Approach

1. **Balanced Representation**: Our method ensures that each pillar contributes meaningfully to the final composite indicator.

2. **Statistical Rigor with Conceptual Validity**: By combining statistical criteria with conceptual importance, we avoid over-reliance on purely mathematical considerations.

3. **Transparency**: Clear documentation of decision criteria enhances reproducibility and trust in the selection process.

4. **Dimensionality Reduction**: Despite maintaining pillar representation, we achieve substantial dimension reduction, improving interpretability.

### Methodological Considerations

1. **Subjectivity in Conceptual Importance**: While we've incorporated conceptual importance in our scoring, this introduces some subjectivity.

2. **Balancing Parsimony and Comprehensiveness**: There is an inherent tension between reducing indicators (parsimony) and maintaining comprehensive representation.

3. **Sensitivity to Thresholds**: Our approach depends on threshold values (e.g., MIN_INDICATORS_PER_PILLAR), which could be adjusted based on specific project needs.

### Implications for Steps 5 and 6

1. **For Normalization (Step 5)**:
   - The selected indicators will require normalization that respects their distributions
   - Special attention should be given to indicators with skewed distributions

2. **For Weighting and Aggregation (Step 6)**:
   - PCA results can inform weighting strategies (e.g., PCA-derived weights)
   - The balanced pillar representation supports meaningful aggregation

## Conclusion

Our implementation of Step 4 provides a robust foundation for constructing a valid and meaningful composite indicator. By improving on traditional indicator selection methods, we've ensured that each conceptual dimension is properly represented while still benefiting from substantial dimension reduction.

The multivariate analysis has revealed important structures in the data, including clear community clusters and interpretable principal components, which enhance our understanding of the relationships between indicators and communities.

The selected indicators are now ready for normalization in Step 5, with a documentation trail that supports transparency and methodological rigor throughout the composite indicator construction process. 