# Step 6: Weighting & Aggregation

## Overview

This step applies weighting and aggregation methods to create the final Community Risk Index (CRI). Following the OECD/JRC Handbook's guidelines, we implement multiple weighting approaches, check for double-counting, and aggregate indicators into pillar scores and then into the final composite index.

## Key Components

### Theoretical Framework

The Community Risk Index is built on four theoretical pillars:
- **Demographics**: Demographic composition and heterogeneity
- **Income**: Economic deprivation and financial strain
- **Housing**: Physical environment, residential stability, and family structure
- **Crime**: Various crime measures as outcome indicators

### Weighting Approaches

Three weighting approaches were implemented:
1. **Equal Weights (EW)**: Simple arithmetic mean
2. **PCA-Variance Weights**: Weights based on explained variance from Principal Component Analysis
3. **Stakeholder Budget Allocation**: Weights based on hypothetical stakeholder priorities

### Double-Counting Check

Correlation analysis identified highly correlated indicators within each pillar. When high correlation (|r| > 0.9) was detected, weights were adjusted to prevent double-counting.

### Aggregation Method

Linear weighted mean was used for aggregation, both within pillars and for the final composite index. This allows for partial compensability between indicators and pillars.

### Sensitivity Analysis

Multiple weighting scenarios were compared to assess the robustness of the index and identify communities most sensitive to methodological choices.

## Directory Structure

- `code/`: Contains the Python script implementing the weighting and aggregation
  - `weighting_aggregation.py`: Main script that performs all calculations
- `docs/`: Contains documentation on the methodology and approach
  - `weighting_aggregation_methodology.md`: Detailed explanation of methods and decisions
- `output/`: Contains the results of the weighting and aggregation
  - `weights_within.csv`: Weights for indicators within each pillar
  - `weights_within_adjusted.csv`: Adjusted weights after correlation check
  - `pillar_scores.csv`: Scores for each pillar
  - `weights_between.csv`: Weights for pillars in the final composite
  - `composite_scores.csv`: Final Community Risk Index scores
  - `composite_scores_ranked.csv`: Ranked communities by CRI
  - `composite_score_summary.csv`: Summary statistics for composite scores
  - `figures/`: Visualizations of results
    - `corr_pillar_*.png`: Correlation heatmaps for each pillar
    - `top_20_communities.png`: Bar chart of top communities
    - `radar_top5.png`: Radar chart of pillar scores for top communities
    - `radar_bottom5.png`: Radar chart of pillar scores for bottom communities
    - `rank_shift_plot.png`: Visualization of rank shifts between weighting scenarios

## How to Run

To execute the weighting and aggregation process:

```bash
python Step_6/code/weighting_aggregation.py
```

This will:
1. Load normalized data from Step 5
2. Calculate weights within pillars
3. Adjust weights for high correlation
4. Aggregate indicators into pillar scores
5. Calculate composite scores using different weighting scenarios
6. Generate visualizations
7. Perform quality checks

## Outputs

The main outputs are:
- **Pillar scores**: Scores for each of the four theoretical pillars
- **Composite scores**: Three versions of the Community Risk Index using different weighting schemes
- **Community rankings**: Rankings of communities based on their CRI scores
- **Visualizations**: Charts showing the results and sensitivity analyses

## Conclusion

Step 6 successfully transforms the normalized indicators into a meaningful composite index that captures the multidimensional nature of community crime risk. The use of multiple weighting scenarios provides flexibility and acknowledges different stakeholder perspectives, while the methodology ensures a transparent and theoretically grounded approach. 