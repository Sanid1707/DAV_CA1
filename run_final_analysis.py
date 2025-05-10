#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Analysis Runner

This script runs both Step 8 (Decomposition & Profiling) and Step 9 (External Linkage & Criterion Validity)
to generate a comprehensive set of visualizations and analysis reports for the CCVEI.
"""

import os
import subprocess
import time
from pathlib import Path

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

def run_step(script_path, step_name):
    """Run a Python script and handle any errors"""
    print_header(f"Running {step_name}")
    start_time = time.time()
    
    try:
        # Run the script using subprocess
        result = subprocess.run(['python', script_path], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        
        # Print the output
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        elapsed_time = time.time() - start_time
        print(f"{step_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running {step_name}:")
        print(e.stderr)
        elapsed_time = time.time() - start_time
        print(f"{step_name} failed after {elapsed_time:.2f} seconds")
        return False

def main():
    """Main function to run all analyses"""
    print_header("COMMUNITY CRIME VULNERABILITY AND EXPOSURE INDEX (CCVEI) - FINAL ANALYSIS")
    
    # Create output directories if they don't exist
    Path("Step_8/output/figures").mkdir(parents=True, exist_ok=True)
    Path("Step_8/output/tables").mkdir(parents=True, exist_ok=True)
    Path("Step_9/output/figures").mkdir(parents=True, exist_ok=True)
    Path("Step_9/output/tables").mkdir(parents=True, exist_ok=True)
    
    # Dictionary of steps to run
    steps = {
        "Step 8: Decomposition & Profiling": "Step_8/code/decomposition.py",
        "Step 9: Basic Visualizations": "Step_9/code/visualization_map.py",
        "Step 9: Geographic Visualizations": "Step_9/code/choropleth_map.py"
    }
    
    # Track success/failure of each step
    results = {}
    
    # Run each step
    for step_name, script_path in steps.items():
        if os.path.exists(script_path):
            result = run_step(script_path, step_name)
            results[step_name] = result
        else:
            print(f"ERROR: Script {script_path} not found.")
            results[step_name] = False
    
    # Summary of results
    print_header("ANALYSIS SUMMARY")
    for step_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{step_name}: {status}")
    
    # Check if all steps were successful
    all_successful = all(results.values())
    if all_successful:
        print("\nAll analyses completed successfully!")
        print("Output files can be found in Step_8/output/ and Step_9/output/ directories.")
    else:
        print("\nSome analyses failed. Please check the errors above.")
    
    return all_successful

if __name__ == "__main__":
    main() 