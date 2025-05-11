#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup Script - Remove redundant files from Step_4/output directory
"""

import os
import shutil

print("\n" + "="*80)
print("CLEANING UP REDUNDANT FILES IN STEP_4/OUTPUT DIRECTORY")
print("="*80)

# Set the output directory path
output_dir = "../output"

# List of redundant files to delete
redundant_files = [
    # Theoretical approach files that have been merged into the main workflow
    "theoretical_analysis_summary.md",
    "theoretical_final_indicators.csv",
    "theoretical_trimmed_dataset.csv",
    
    # Outdated files based on previous statistical approach
    "final_indicators_summary.csv",
    "final_indicators_pivot.csv",
    "indicator_decision_report.md",
    "pca_comparison.csv",
    
    # Redundant figures
    "figures/theoretical_scree_plot_Crime.png",
    "figures/theoretical_scree_plot_Housing.png",
    "figures/theoretical_scree_plot_Income.png",
    "figures/theoretical_scree_plot_Demographics.png"
]

# Files deleted counter
deleted_count = 0

# Delete each redundant file
for file_path in redundant_files:
    full_path = os.path.join(output_dir, file_path)
    if os.path.exists(full_path):
        try:
            if os.path.isfile(full_path):
                os.remove(full_path)
                print(f"Deleted: {file_path}")
                deleted_count += 1
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(f"Deleted directory: {file_path}")
                deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

print(f"\nCleanup complete. Deleted {deleted_count} redundant files.")

# Create a more organized directory structure
print("\nOrganizing files into subdirectories...")

# Directories to create
directories = [
    "docs",            # For documentation and reports
    "data",            # For datasets
    "indicators",      # For indicator selection files
    "pca_results",     # For PCA analysis results
    "cluster_results"  # For clustering results
]

# Create directories if they don't exist
for directory in directories:
    dir_path = os.path.join(output_dir, directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {directory}")

# File mapping for reorganization
file_mapping = {
    # Documentation files
    "step4_analysis_summary.md": "docs/step4_analysis_summary.md",
    "theoretical_selection_approach.md": "docs/theoretical_selection_approach.md",
    "step4_comprehensive_guide.md": "docs/step4_comprehensive_guide.md",
    "executive_summary.md": "docs/executive_summary.md",
    "key_findings_reference.md": "docs/key_findings_reference.md",
    "visualization_guide.md": "docs/visualization_guide.md",
    "component_selection_rationale.txt": "docs/component_selection_rationale.txt",
    
    # Dataset files
    "step4_trimmed_dataset.csv": "data/step4_trimmed_dataset.csv",
    "step4_final_dataset.csv": "data/step4_final_dataset.csv",
    
    # Indicator selection files
    "final_indicators.csv": "indicators/final_indicators.csv",
    "indicator_decision_grid.csv": "indicators/indicator_decision_grid.csv",
    
    # Clustering results
    "all_clusters_summary.md": "cluster_results/all_clusters_summary.md",
    "Crime_cluster_description.md": "cluster_results/Crime_cluster_description.md",
    "Crime_cluster_profiles.csv": "cluster_results/Crime_cluster_profiles.csv",
    "Housing_cluster_description.md": "cluster_results/Housing_cluster_description.md",
    "Housing_cluster_profiles.csv": "cluster_results/Housing_cluster_profiles.csv",
    "Income_cluster_description.md": "cluster_results/Income_cluster_description.md",
    "Income_cluster_profiles.csv": "cluster_results/Income_cluster_profiles.csv",
    "Demographics_cluster_description.md": "cluster_results/Demographics_cluster_description.md",
    "Demographics_cluster_profiles.csv": "cluster_results/Demographics_cluster_profiles.csv",
    
    # Correlation matrices 
    "Crime_correlation_matrix.csv": "pca_results/Crime_correlation_matrix.csv",
    "Housing_correlation_matrix.csv": "pca_results/Housing_correlation_matrix.csv",
    "Income_correlation_matrix.csv": "pca_results/Income_correlation_matrix.csv",
    "Demographics_correlation_matrix.csv": "pca_results/Demographics_correlation_matrix.csv"
}

# Move files to their new locations
moved_count = 0
for source, destination in file_mapping.items():
    source_path = os.path.join(output_dir, source)
    dest_path = os.path.join(output_dir, destination)
    
    if os.path.exists(source_path):
        try:
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Move the file
            shutil.move(source_path, dest_path)
            print(f"Moved: {source} → {destination}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving {source}: {e}")
    else:
        print(f"Source file not found: {source}")

print(f"\nReorganization complete. Moved {moved_count} files to appropriate subdirectories.")
print("\nOutput directory cleanup and organization complete!") 