
## Method Rationale for Data Imputation

### B1: Missingness Mechanism Assumption
We assume that data are Missing At Random (MAR), meaning the probability of a value being missing depends on observed data but not on the missing data itself. This is a reasonable assumption for crime datasets, where missingness in socioeconomic indicators may be related to other observable variables like population size or location, but not directly to the missing values themselves.

### B2: Strategy for Percentage Columns (0-100)
For percentage columns, we use predictive mean matching to ensure imputed values stay within the logical 0-100 range. This method borrows values from observed data points with similar predicted means, preserving the distribution characteristics of percentage variables.

### B3: Strategy for Counts and Monetary Columns
- For count variables: We use an iterative imputation approach with a Random Forest regressor as the estimation model, which handles the skewed nature of count data well. For imputed values, we round to the nearest integer and enforce non-negativity.
- For monetary variables: We use the same Random Forest based iterative imputation to handle the typically skewed distributions of monetary values, with non-negativity constraints.

### B4: Columns with ≤ 5% Missing
For columns with 5% or fewer missing values, we use median imputation for skewed variables and mean imputation for more symmetric distributions. This approach is computationally efficient and provides reliable estimates when missingness is low.

### B5: Columns with > 5% Missing
For columns with more than 5% missing values and for strategically important variables (crime rates), we implement multiple imputation using 5 iterations to capture uncertainty. The results are combined to provide both point estimates and insights into the variability introduced by imputation.

### Additional Considerations
- All imputation methods preserve the variable's distribution characteristics to prevent distortion of relationships.
- For variables with strong correlations, we leverage multivariate relationships through the iterative imputation process.
- The imputation process is documented in detail for transparency and reproducibility.
