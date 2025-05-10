# Implementation Guidance: Adapting the Project to the CCVEI Framework

This document provides guidance on how to adapt the existing implementation to align with the new Community Crime Vulnerability and Exposure Index (CCVEI) theoretical framework without requiring significant code changes.

## Maintaining Existing Implementation

The good news is that the current implementation can be maintained almost entirely as-is, with only documentation and interpretation changes needed to align with the new framework. This is because:

1. The four pillars (Demographics, Income, Housing, Crime) remain the same
2. The indicators and data sources remain unchanged
3. The weighting and aggregation methodologies remain valid
4. The visualization approaches continue to be relevant

## Required Documentation Changes

The following documentation files should be updated to reflect the new framework:

1. **README files**: Update all README files to use the new CCVEI name and framework description
2. **Validation Report**: Revise to reflect the dual vulnerability/exposure approach
3. **Step descriptions**: Update conceptual descriptions in each step document

## Key Terminology Changes

Apply these terminology changes consistently across all documentation:

| Original Term | New Term |
|--------------|----------|
| Community Crime Risk Index (CCRI) | Community Crime Vulnerability and Exposure Index (CCVEI) |
| Crime risk prediction | Crime vulnerability and exposure assessment |
| Risk factors | Vulnerability and exposure factors |
| Risk profile | Vulnerability and exposure profile |

## Interpretation Adjustments

When interpreting results in the documentation:

1. **For Demographics, Income, and Housing pillars**: Describe as "vulnerability factors" that affect a community's resilience to crime
2. **For the Crime pillar**: Describe as "exposure factors" that represent current crime conditions
3. **For overall index scores**: Describe as representing the combined assessment of vulnerability and exposure

## Visualization Framework Alignment

Current visualizations can be maintained with these interpretation adjustments:

1. **Distribution plots**: Present as showing the distribution of vulnerability and exposure profiles
2. **Top/bottom communities**: Frame as communities with highest/lowest combined vulnerability and exposure
3. **Pillar comparisons**: Emphasize the distinction between vulnerability pillars and the exposure pillar
4. **Geographic visualizations**: Describe as showing spatial patterns of vulnerability and exposure

## Code Adjustments (Minimal)

Very few code changes are required, primarily variable naming and comments:

1. **Variable names**: If possible, rename key variables (e.g., `ccri_score` to `ccvei_score`)
2. **Function documentation**: Update docstrings to reflect the new framework
3. **Output file names**: Consider updating output filenames to reflect CCVEI terminology

## Theoretical Integration in Step 9

For Step 9 (External Linkage & Criterion Validity), enhance the theoretical justification:

1. **Validation approach**: Explicitly validate both the vulnerability components and exposure components separately
2. **Framework justification**: Add discussion of how the integration of vulnerability and exposure provides a more comprehensive assessment than either alone
3. **Geographic patterns**: Interpret geographic patterns in terms of both vulnerability factors and crime exposure

## Communication Guidelines

When presenting the index to stakeholders:

1. Be explicit about the dual-framework approach
2. Emphasize that this is not attempting to "predict" crime using crime data (avoiding the circular reference)
3. Highlight the value of assessing both current exposure and underlying vulnerability
4. Stress how this approach aligns with established risk assessment frameworks in other fields (e.g., disaster management)

## Implementation Checklist

- [ ] Update all README files with new terminology
- [ ] Revise validation report to align with new framework
- [ ] Update variable names and docstrings in code (where feasible)
- [ ] Adjust output file naming conventions (optional)
- [ ] Create presentation materials explaining the framework
- [ ] Document the theoretical justification for the framework shift
- [ ] Add explanation of how the framework addresses the circular reference concern 