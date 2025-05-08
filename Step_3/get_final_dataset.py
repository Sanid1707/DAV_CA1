#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 Final Dataset Access Helper
- This script extracts the final Step 3 dataset for use in subsequent analysis steps
- Simply run this script to copy the final dataset to the main project directory
"""

import os
import shutil
import pandas as pd

print("\nStep 3: Final Dataset Access Tool")
print("="*50)

# Define paths
source_file = os.path.join("data", "step3_outlier_handled_dataset.csv")
target_file = os.path.join("..", "step3_final_dataset_for_multivariate.csv")

# Check if the source file exists
if not os.path.exists(source_file):
    print(f"Error: Source file '{source_file}' not found.")
    print("Make sure you are running this script from within the Step_3 directory.")
    exit(1)

# Copy the file to the main directory
try:
    shutil.copy2(source_file, target_file)
    
    # Verify the copy was successful
    if os.path.exists(target_file):
        # Load the data to show a sample
        df = pd.read_csv(target_file)
        
        print(f"\nSuccess! Final dataset copied to: {os.path.abspath(target_file)}")
        print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        print("\nThis dataset has been prepared with:")
        print("- Complete data (no missing values)")
        print("- Outlier handling specific to each variable type")
        print("- Transform application as appropriate")
        print("\nThe dataset is now ready for multivariate analysis in Step 4.")
    else:
        print("Error: File copy verification failed.")
except Exception as e:
    print(f"Error copying the file: {e}") 