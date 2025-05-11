#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Multivariate Analysis - Main Runner Script
This script runs all the modules of Step 4 in sequence:
1. Main PCA and clustering (Step_4.py)
2. Cluster profiling (Step_4_cluster_profiling.py)
3. Collinearity analysis (Step_4_collinearity.py)
4. Indicator selection and final processing (Step_4_indicator_selection.py)
"""

import os
import subprocess
import time

# Create necessary directories
os.makedirs('Step_4/data', exist_ok=True)
os.makedirs('Step_4/output', exist_ok=True)
os.makedirs('Step_4/output/figures', exist_ok=True)

print("\n" + "="*80)
print("STEP 4: COMPLETE MULTIVARIATE ANALYSIS PROCESS")
print("="*80)

# Function to run a Python script and capture its output
def run_script(script_path):
    """Run a Python script and print its output in real-time"""
    print(f"\nRunning {script_path}...\n" + "-"*80)
    
    # Run the script as a subprocess
    process = subprocess.Popen(['python', script_path], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.STDOUT,
                              universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        print(f"Error running {script_path}. Return code: {process.returncode}")
        return False
    
    print("-"*80 + f"\nCompleted {script_path}\n")
    return True

# Record start time
start_time = time.time()

# Run Step 4 modules in sequence
scripts = [
    'Step_4/code/Step_4.py',                    # Main PCA and clustering
    'Step_4/code/Step_4_cluster_profiling.py',  # Cluster profiling
    'Step_4/code/Step_4_collinearity.py',       # Collinearity analysis
    'Step_4/code/Step_4_indicator_selection.py' # Indicator selection and final processing
]

success = True
for script in scripts:
    if not run_script(script):
        success = False
        print(f"Error running {script}. Stopping execution.")
        break
    time.sleep(1)  # Small pause between scripts

# Calculate and print execution time
end_time = time.time()
execution_time = end_time - start_time
minutes, seconds = divmod(execution_time, 60)
hours, minutes = divmod(minutes, 60)

print("\n" + "="*80)
if success:
    print(f"STEP 4 COMPLETED SUCCESSFULLY in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nResults are available in the Step_4/output directory")
    print("The trimmed dataset is ready for Step 5 normalization")
else:
    print("STEP 4 EXECUTION FAILED")
    print("Please check the error messages above and fix any issues")
print("="*80) 