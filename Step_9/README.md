# Step 9: External Linkage & Criterion Validity

This step validates the Community Crime Vulnerability and Exposure Index (CCVEI) by analyzing its distribution, examining top and bottom communities, and comparing different pillars and weighting methods.

## Overview

The analysis in Step 9 demonstrates criterion/convergent validity by examining how the CCVEI relates to its component pillars (both vulnerability factors and exposure factors) and by comparing different weighting methodologies. This approach provides evidence of the index's robustness and allows for identification of communities at the extremes of the vulnerability and exposure spectrum.

## Visualizations

The following visualizations have been generated:

1. **CCVEI Distribution** (`ccri_distribution.png`): Shows the statistical distribution of vulnerability and exposure profiles across all communities, with mean and median lines.

2. **Top Communities by Vulnerability and Exposure** (`top_communities_by_risk.png`): Bar charts displaying the 10 highest and 10 lowest communities based on their combined vulnerability and exposure scores.

3. **Pillar Comparison** (`pillar_comparison.png`): Shows the average scores across the four main pillars that make up the CCVEI:
   - Demographics (Vulnerability Factor)
   - Income (Vulnerability Factor)
   - Housing (Vulnerability Factor)
   - Crime (Exposure Factor)

4. **Pillar Correlation Heatmap** (`pillar_correlation.png`): Displays the correlation coefficients between the vulnerability pillars and the exposure pillar, revealing relationships between different components.

5. **Weighting Method Comparison** (`weighting_comparison_means.png`): Compares the average CCVEI scores using three different weighting methods:
   - Equal Weights
   - PCA Weights
   - Stakeholder Weights

6. **Weighting Method Scatter Plot Matrix** (`weighting_comparison_scatter.png`): Shows how the different weighting methodologies relate to each other through pairwise scatter plots.

7. **Geographic Visualizations**:
   - **Interactive State Map** (`ccri_choropleth_interactive.html`): Interactive visualization showing CCVEI scores by state
   - **State Comparison Chart** (`state_ccri_comparison.png`): Bar chart comparing average CCVEI scores across states
   - **State Heatmap** (`state_ccri_heatmap.png`): Heatmap visualization of vulnerability and exposure patterns by state

## Data Tables

The analysis also generates several CSV files containing key findings:

1. **Highest Vulnerability & Exposure Communities** (`highest_risk_communities.csv`): Lists the top 10 communities with the highest combined vulnerability and exposure scores.

2. **Lowest Vulnerability & Exposure Communities** (`lowest_risk_communities.csv`): Lists the 10 communities with the lowest combined vulnerability and exposure scores.

3. **Weighting Method Statistics** (`weighting_method_statistics.csv`): Contains summary statistics (mean, median, standard deviation, min, max) for each weighting method.

## Validation Approach

This implementation validates the CCVEI by:

1. **Internal consistency**: Examining correlations between vulnerability pillars and the exposure pillar to ensure they contribute meaningfully but distinctly to the overall index.

2. **Weighting robustness**: Analyzing how different weighting approaches affect the final scores, providing evidence of methodological stability.

3. **Extremes analysis**: Identifying and documenting communities at the highest and lowest vulnerability and exposure levels, which can serve as case studies for intervention or emulation.

4. **Geographic validation**: Visualizing the spatial distribution of vulnerability and exposure patterns to identify regional patterns and validate against known criminological geographic trends.

## Theoretical Relevance

The visualizations and data produced in this step align with our integrated theoretical framework:

1. **Social Vulnerability Theory**: Demonstrating how demographic, economic, and housing factors create differential vulnerability to crime impacts across communities.

2. **Economic Strain Theory**: Showing the relationship between economic factors and overall community vulnerability and exposure profiles.

3. **Social Disorganization Theory**: Highlighting communities where social and physical environment factors combine with current crime exposure to create challenging conditions.

4. **Crime Pattern Theory**: Revealing how current crime exposure patterns relate to underlying vulnerability factors and vary geographically.

## Running the Code

To run the complete analysis:

```bash
python run_final_analysis.py
```

This will run both Step 8 and Step 9 scripts sequentially.

To run just the Step 9 visualizations:

```bash
python Step_9/code/visualization_map.py
python Step_9/code/choropleth_map.py
```

The scripts will generate all visualizations and data tables in the `Step_9/output/` directory.

## Requirements

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- plotly 