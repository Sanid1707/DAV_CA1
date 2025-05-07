# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#############################################################
# SECTION 1: SETUP AND INITIALIZATION
#############################################################
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from statsmodels.stats.outliers_influence import variance_inflation_factor
import networkx as nx
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Set style for the plots - Using a more modern Seaborn style
plt.style.use('seaborn-v0_8-whitegrid') # Cleaner style
sns.set_palette('viridis') # Perceptually uniform categorical palette
plt.rcParams['figure.figsize'] = (16, 10) # Default figure size
plt.rcParams['font.size'] = 12 # Default font size
plt.rcParams['axes.titlesize'] = 16 # Title font size
plt.rcParams['axes.labelsize'] = 14 # Axis label font size
plt.rcParams['xtick.labelsize'] = 12 # X-tick label size
plt.rcParams['ytick.labelsize'] = 12 # Y-tick label size
plt.rcParams['legend.fontsize'] = 12 # Legend font size
plt.rcParams['figure.dpi'] = 100 # Default DPI for screen
plt.rcParams['savefig.dpi'] = 300 # Higher DPI for saved figures

# Try different encodings to load the dataset
try:
    # Try with Latin-1 encoding
    df = pd.read_csv('crimedata.csv', encoding='latin1')
except:
    try:
        # Try with ISO-8859-1 encoding
        df = pd.read_csv('crimedata.csv', encoding='ISO-8859-1')
    except:
        try:
            # Try with Windows-1252 encoding
            df = pd.read_csv('crimedata.csv', encoding='cp1252')
        except:
            # Fallback to detect encoding
            import chardet
            with open('crimedata.csv', 'rb') as file:
                result = chardet.detect(file.read())
            df = pd.read_csv('crimedata.csv', encoding=result['encoding'])

