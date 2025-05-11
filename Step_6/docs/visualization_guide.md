# Visualization Guide for Step 6: Weighting & Aggregation

This document explains what each visualization in the `Step_6/output/figures` directory represents and how to interpret it.

## Correlation Matrices

### Individual Pillar Correlation Matrices
Files: `enhanced_corr_Demographics.png`, `enhanced_corr_Income.png`, `enhanced_corr_Housing.png`, `enhanced_corr_Crime.png`

**What they show**: 
- These visualizations display the correlation coefficients between indicators within each pillar.
- Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).
- The white triangular area in the upper right is intentional, showing only the lower triangular part of the symmetric correlation matrix to avoid redundancy.
- Orange annotations highlight highly correlated pairs (|r| > 0.9) that might lead to double-counting.

**Purpose**:
- Identify potential double-counting issues within each pillar
- Guide weight adjustments where indicators are highly correlated
- Understand relationships between indicators in the same theoretical domain

**Note on Grid Alignment**: 
You may notice slight misalignments between the grid lines and cells in some correlation matrices. This is a common issue with heatmap visualizations when the figure dimensions don't perfectly match the number of variables. To fix this in future iterations, we can modify the code to:
```python
plt.figure(figsize=(10, 8))
ax = plt.gca()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            mask=mask, fmt='.2f', linewidths=0.5, square=True, ax=ax)
ax.set_aspect('equal')  # This ensures perfect grid alignment
```

### Combined Correlation Matrices
File: `all_correlation_matrices.png`

**What it shows**:
- A 2×2 grid displaying all four pillar correlation matrices in a single visualization for easy comparison.

**Purpose**:
- Allow side-by-side comparison of correlation patterns across different pillars
- Identify which pillars have more internal correlation issues

## Radar Charts

### Individual Pillar Score Radar Charts
Files: `enhanced_radar_top5.png`, `improved_radar_top5.png`, `enhanced_radar_bottom5.png`, `improved_radar_bottom5.png`

**What they show**:
- Pillar scores for the top 5 and bottom 5 communities across all four pillars.
- Each axis represents one pillar (Demographics, Income, Housing, Crime).
- Each colored line represents a different community.

**Purpose**:
- Visualize the multi-dimensional performance profile of communities
- Identify whether top/bottom communities are balanced across pillars or excel/fail in specific areas
- Compare the shape of performance profiles between high and low scoring communities

### Combined Top vs Bottom Radar Charts
Files: `top_vs_bottom_radar.png`, `combined_radar_chart.png`

**What they show**:
- Side-by-side comparison of the top 5 and bottom 5 communities on the same pillar dimensions.

**Purpose**:
- Directly compare the performance profiles of the best and worst-performing communities
- Identify which pillars show the greatest differentiation between top and bottom communities

## Rank Shift Analysis

### Rank Shift Plots
Files: `enhanced_rank_shift_pca.png`, `improved_rank_shift_plot.png`, `enhanced_rank_shift_stakeholder.png`, `improved_rank_shift_stakeholder.png`

**What they show**:
- Communities with the largest changes in rank position when different weighting schemes are applied.
- Horizontal lines connect a community's rank under one weighting scheme to its rank under another.
- The length of each line represents the magnitude of rank change.
- Annotations show the numerical rank difference (Δ).

**Purpose**:
- Identify communities most sensitive to methodological choices
- Assess the robustness of the Community Risk Index rankings
- Demonstrate the impact of different normative judgments (equal vs. PCA vs. stakeholder weights)

## Bar Charts

### Top Communities Bar Charts
Files: `enhanced_top_15_communities.png`, `enhanced_top_20_communities.png`

**What they show**:
- Horizontal bar charts of the top 15/20 communities by their composite CRI score.
- Bars are colored using a gradient for visual appeal.
- Value labels show the precise CRI score for each community.

**Purpose**:
- Clearly communicate the highest-performing communities
- Show the distribution and gaps in scores among top performers
- Provide an accessible visualization for non-technical audiences

## Notes on Data Source

All visualizations were created using the normalized dataset from Step 5 (`Step_5/output/step5_normalized_dataset.csv`). This ensures that all indicators were on the same scale (0-1) before applying weights and aggregation, as required by the OECD methodology.

## Interpretation Guidelines

When interpreting these visualizations:

1. **Correlation Matrices**: Higher absolute values (deeper reds or blues) indicate stronger relationships between indicators. Pay special attention to very high correlations (> 0.9) as they suggest potential redundancy.

2. **Radar Charts**: The further a line extends outward on each axis, the better the community performs in that pillar. Perfectly balanced communities would appear as regular polygons.

3. **Rank Shift Plots**: Longer horizontal lines indicate communities whose ranking is highly sensitive to the weighting scheme chosen. These communities typically have uneven performance across pillars.

4. **Bar Charts**: These show absolute values rather than ranks, providing a better sense of the actual performance gaps between communities.

These visualizations should be considered together to form a comprehensive understanding of both the Community Risk Index results and the methodological choices that influenced those results. 