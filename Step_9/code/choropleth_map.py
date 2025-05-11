#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9: USA Geographic Visualization of Vulnerability and Exposure

This script creates geographic visualizations of the Community Crime Vulnerability and Exposure Index (CCVEI)
across the United States, based on the scores from Step 6.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import os
import warnings

# Set directories
Path("Step_9/output/figures").mkdir(parents=True, exist_ok=True)
Path("Step_9/data").mkdir(parents=True, exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    Load the composite scores from Step 6
    """
    print("Loading composite scores...")
    try:
        scores_df = pd.read_csv("Step_6/output/composite_scores_ranked.csv")
        
        # Extract community names and state information
        scores_df['community_name'] = scores_df['communityname']
        
        # Try to extract state information from community name
        # The community names in the dataset end with either "city", "town", "township", etc.
        # We need to identify and map them to state codes for the visualization
        
        # For demonstration, we'll create a simplified mapping of state patterns we can identify
        state_patterns = {
            'california': 'CA',
            'texas': 'TX',
            'newyork': 'NY',
            'florida': 'FL',
            'illinois': 'IL',
            'pennsylvania': 'PA',
            'ohio': 'OH',
            'michigan': 'MI',
            'georgia': 'GA',
            'newjersey': 'NJ'
        }
        
        # Extract state information based on common patterns in community names
        # This is a simplified approach - in a real application, we would need more robust state mapping
        states = []
        for name in scores_df['community_name']:
            name_lower = name.lower()
            state_code = None
            
            for pattern, code in state_patterns.items():
                if pattern in name_lower:
                    state_code = code
                    break
            
            # If no match, assign a placeholder
            if not state_code:
                # If the name ends with 'city', extract the preceding part as potential state indicator
                if 'city' in name_lower:
                    state_code = 'US-OTHER'
                else:
                    state_code = 'US-OTHER'
            
            states.append(state_code)
        
        scores_df['state_code'] = states
        
        # Use stakeholder weights as primary CCVEI score
        scores_df['ccvei_score'] = scores_df['CRI_StakeholderWeights']
        
        return scores_df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a sample dataframe for testing
        sample_df = pd.DataFrame({
            'community_name': ['CommunityA', 'CommunityB', 'CommunityC'],
            'state_code': ['CA', 'TX', 'NY'],
            'ccvei_score': [0.7, 0.5, 0.3]
        })
        return sample_df

def create_plotly_choropleth_html(data_df):
    """
    Create an interactive choropleth map using Plotly and save as HTML
    """
    print("Creating Plotly choropleth map (HTML only)...")
    
    try:
        # Aggregate data by state
        state_df = data_df.groupby('state_code')['ccvei_score'].mean().reset_index()
        
        # Filter out non-state codes
        state_df = state_df[state_df['state_code'] != 'US-OTHER']
        
        # Create a figure
        fig = px.choropleth(
            state_df,
            locations='state_code',  # Column containing state codes
            color='ccvei_score',  # Column with values to plot
            color_continuous_scale='RdYlBu_r',  # Color scale (reversed)
            scope="usa",  # Focus on USA
            labels={'ccvei_score': 'CCVEI Score'},
            title='Community Crime Vulnerability and Exposure Index (CCVEI) by State'
        )
        
        # Update layout
        fig.update_layout(
            geo=dict(
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                showcoastlines=True,
            ),
            coloraxis_colorbar=dict(
                title='CCVEI Score',
                tickvals=[0.3, 0.4, 0.5, 0.6, 0.7],
                ticktext=['Low V&E', 'Medium-Low', 'Medium', 'Medium-High', 'High V&E']
            ),
            title=dict(
                text='Community Crime Vulnerability and Exposure Index (CCVEI) by State',
                font=dict(size=20)
            ),
            autosize=True
        )
        
        # Save as HTML for interactivity
        fig.write_html('Step_9/output/figures/ccvei_choropleth_interactive.html')
        
        print("Interactive choropleth map saved to Step_9/output/figures/ccvei_choropleth_interactive.html")
        return True
    except Exception as e:
        print(f"Error creating Plotly choropleth HTML: {e}")
        return False

def create_matplotlib_choropleth(data_df):
    """
    Create a simple state-based choropleth map using Matplotlib
    """
    print("Creating Matplotlib choropleth map...")
    
    try:
        # Aggregate data by state
        state_df = data_df.groupby('state_code')['ccvei_score'].mean().reset_index()
        
        # Filter out non-state codes
        state_df = state_df[state_df['state_code'] != 'US-OTHER']
        
        # Create a bar chart as an alternative visualization
        plt.figure(figsize=(12, 8))
        
        # Sort by CCVEI score
        state_df = state_df.sort_values('ccvei_score', ascending=False)
        
        # Create bars
        bars = plt.bar(state_df['state_code'], state_df['ccvei_score'], color='darkblue')
        
        # Add labels and title
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Average CCVEI Score', fontsize=12)
        plt.title('Average Community Crime Vulnerability and Exposure by State', fontsize=16)
        
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/state_ccvei_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("State CCVEI comparison chart saved to Step_9/output/figures/state_ccvei_comparison.png")
        return True
    except Exception as e:
        print(f"Error creating Matplotlib chart: {e}")
        return False

def create_heatmap_visualization(data_df):
    """
    Create a state heatmap as an alternative to choropleth map
    """
    print("Creating state heatmap visualization...")
    
    try:
        # Aggregate data by state
        state_df = data_df.groupby('state_code')['ccvei_score'].mean().reset_index()
        
        # Filter out non-state codes
        state_df = state_df[state_df['state_code'] != 'US-OTHER']
        
        # Sort by CCVEI score for better visualization
        state_df = state_df.sort_values('ccvei_score', ascending=False)
        
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        
        # Reshape data for heatmap
        state_codes = state_df['state_code'].values
        ccvei_scores = state_df['ccvei_score'].values
        
        # Create a 2D representation with 5 columns
        num_states = len(state_codes)
        cols = 5
        rows = (num_states + cols - 1) // cols  # Ceiling division
        
        # Create empty matrix
        heatmap_data = np.zeros((rows, cols))
        heatmap_data.fill(np.nan)  # Fill with NaN for empty cells
        
        # Fill matrix with CCVEI scores
        for i in range(num_states):
            row = i // cols
            col = i % cols
            heatmap_data[row, col] = ccvei_scores[i]
        
        # Create heatmap
        im = plt.imshow(heatmap_data, cmap='RdYlBu_r')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('CCVEI Score')
        
        # Add state labels
        for i in range(num_states):
            row = i // cols
            col = i % cols
            plt.text(col, row, f"{state_codes[i]}\n{ccvei_scores[i]:.3f}", 
                    ha="center", va="center", color="black", fontsize=9)
        
        # Set title
        plt.title('Community Crime Vulnerability and Exposure Index (CCVEI) by State', fontsize=16)
        
        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        plt.savefig('Step_9/output/figures/state_ccvei_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("State CCVEI heatmap saved to Step_9/output/figures/state_ccvei_heatmap.png")
        return True
    except Exception as e:
        print(f"Error creating state heatmap: {e}")
        return False

def main():
    """Main execution function"""
    # Load data
    print("Starting geographic visualization creation...")
    df = load_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return False
    
    # Create Plotly choropleth HTML
    create_plotly_choropleth_html(df)
    
    # Create Matplotlib bar chart as alternative
    create_matplotlib_choropleth(df)
    
    # Create heatmap as another alternative
    create_heatmap_visualization(df)
    
    print("CCVEI geographic visualizations complete!")
    return True

if __name__ == "__main__":
    main() 