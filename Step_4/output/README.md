# Step 4 Output Directory Structure

This directory contains all outputs from Step 4 of the Community Crime-Risk Index (CCRI) project, organized into subdirectories by type.

## Directory Structure

### `/docs`
Documentation and reports related to the multivariate analysis:
- `step4_analysis_summary.md`: Main summary of the theoretical approach to variable selection
- `theoretical_selection_approach.md`: Detailed explanation of the theoretical selection methodology
- `step4_comprehensive_guide.md`: Comprehensive guide to all Step 4 processes
- `executive_summary.md`: Executive summary of key findings
- `key_findings_reference.md`: Reference for all key findings
- `visualization_guide.md`: Guide to all visualizations created in Step 4
- `component_selection_rationale.txt`: Rationale for principal component selection

### `/data`
Datasets produced during the multivariate analysis:
- `step4_trimmed_dataset.csv`: Final dataset with theoretically selected variables (18 indicators)
- `step4_final_dataset.csv`: Original processed dataset before variable reduction

### `/indicators`
Files related to indicator selection and documentation:
- `final_indicators.csv`: Final list of selected indicators with rationales
- `indicator_decision_grid.csv`: Grid used for decision-making on variable selection

### `/pca_results`
Principal Component Analysis results:
- `*_correlation_matrix.csv`: Correlation matrices for each pillar

### `/cluster_results`
Clustering analysis results:
- `all_clusters_summary.md`: Summary of clustering results across all pillars
- `*_cluster_description.md`: Detailed descriptions of clusters for each pillar
- `*_cluster_profiles.csv`: Cluster profiles and properties

### `/figures`
Visualizations and plots from various analyses:
- PCA visualizations
- Correlation heatmaps
- Dendrograms
- Cluster visualizations
- Scree plots

## Notes on Theoretical Selection Approach

Our variable selection process prioritizes theoretical importance over purely statistical considerations. We've selected 18 theoretically important variables across four pillars:

1. **Demographics Pillar**: 4 indicators
2. **Income Pillar**: 4 indicators 
3. **Housing Pillar**: 5 indicators
4. **Crime Pillar**: 5 indicators

This approach ensures that our composite index:
- Maintains strong theoretical validity
- Captures all key dimensions of crime risk
- Provides sufficient variables for meaningful weighting and aggregation in Steps 5 and 6

## Recent Updates

We've updated our collinearity threshold from 0.9 to 0.95, meaning variables are only considered highly correlated if their correlation coefficient exceeds 0.95 (instead of 0.9). This more permissive threshold allows us to retain more theoretically important variables that might have moderate to high (but not extremely high) correlations.

With this change, fewer variables are flagged for potential removal due to collinearity, providing more flexibility in our theoretical selection approach.

For full details, see `/docs/theoretical_selection_approach.md`. 