# Step 8: Decomposition & Profiling Analysis

This step provides additional analysis to strengthen the theoretical framework of our Community Risk Index by examining how different components contribute to the final scores and validating the relationships between pillars.

## Components

1. **Pillar Contribution Analysis**
   - Visualizes how each pillar contributes to final scores
   - Validates the weighting scheme
   - Located in: `output/figures/pillar_contributions.png`

2. **Community Profiles**
   - Interactive radar plots of top/bottom communities
   - Detailed indicator-level analysis
   - Located in: `output/figures/radar_top5.html` and `radar_bottom5.html`

3. **Correlation Analysis**
   - Examines relationships between pillars and final scores
   - Validates theoretical framework
   - Located in: `output/figures/pillar_correlations.png`

4. **Narrative Profiles**
   - Detailed analysis of highest/lowest risk communities
   - Links findings to theoretical framework
   - Located in: `docs/community_profiles.md`
a
## Directory Structure

```
Step_8/
├── code/
│   └── decomposition_analysis.py
├── docs/
│   ├── analysis_summary.md
│   └── community_profiles.md
├── output/
│   └── figures/
│       ├── pillar_contributions.png
│       ├── pillar_correlations.png
│       ├── radar_top5.html
│       └── radar_bottom5.html
├── README.md
└── requirements.txt
```

## Running the Analysis

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python code/decomposition_analysis.py
   ```

## Theoretical Framework Support

This analysis strengthens our theoretical framework by:

1. **Validating Relationships**
   - Confirms expected correlations between pillars
   - Shows how indicators cluster in high/low risk communities

2. **Supporting Theory**
   - Demonstrates alignment with Social Disorganization Theory
   - Shows evidence for Routine Activity Theory principles

3. **Practical Implications**
   - Identifies key characteristics of high/low risk communities
   - Provides evidence-based insights for policy recommendations

## Integration with Previous Steps

This analysis builds on:
- Normalized data from Step 5
- Weighted scores from Step 6
- Theoretical foundations established throughout the project

The results provide additional validation of our methodological choices and strengthen the theoretical underpinnings of the Community Risk Index. 