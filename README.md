# DAV_CA1

Data Analysis and Visualization Course Assignment 1. This repository contains Jupyter notebooks for data analysis. 

## Crime Variable Correlation Analysis

The analysis of crime data revealed important insights about correlation patterns:

1. **High Correlations Between Crime Count Variables**: Crime count variables naturally have very high correlations (average 0.9154) due to the "size effect" - larger communities simply have more of all crime types because they have more people.

2. **Proper Handling in Data Selection**: Step_2.py already correctly handles correlations within pillars (lines 501-539) and preserves Crime Indicator variables even when highly correlated.

3. **Statistical Phenomenon**: This is a known statistical phenomenon rather than a data selection problem - when using raw counts rather than rates, the underlying population size creates these strong correlations.

4. **More Meaningful Rate-Based Analysis**: The more meaningful analysis comes from crime rate variables (per population), which show more distinct patterns (average correlation 0.5327) that better reflect actual relationships between crime types.

5. **Methodologically Sound Approach**: The current approach is methodologically sound, and the correlation patterns observed reflect the nature of the data rather than any issues with the code. 