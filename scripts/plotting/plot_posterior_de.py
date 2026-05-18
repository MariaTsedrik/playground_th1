#!/usr/bin/env python3
"""
Posterior Distribution Plotting Script for Cosmological Parameters

This script creates triangle (corner) plots of posterior distributions from MCMC chains
for cosmological parameter analysis. It uses GetDist for plotting of
parameter constraints and correlations.

Dependencies:
    - numpy: numerical computations
    - getdist: cosmological parameter plotting and analysis
    - matplotlib: plotting backend
    - yaml: configuration file parsing
    - seaborn: enhanced color palettes

Configuration Files Required:
    - params_names.yaml: parameter name mappings for plot labels
    - params_model_shear.yaml: fiducial parameter values and ranges

Author: Maria Tsedrik
Date: 26 Aug 2025    
"""

import numpy as np
import getdist.plots
import matplotlib.pyplot as plt
import yaml
import sys
import os

# Load parameter configuration files
# params_names.yaml contains the mapping from parameter names to LaTeX labels
with open("params_names.yaml", "r") as file:
    params_dic = yaml.safe_load(file)

# params_model_shear.yaml contains fiducial values and prior ranges
with open("params_model_shear.yaml", "r") as file:
    fiducials = yaml.safe_load(file)



def read_last_header_line(file_path):
    """
    Extract parameter names from the last header line of a chain file.
    
    MCMC chain files typically have header lines starting with '#' that contain
    parameter names. This function finds the last such header line and extracts
    the parameter names from it.
    
    Parameters:
    -----------
    file_path : str
        Path to the MCMC chain file
        
    Returns:
    --------
    list
        List of parameter names extracted from the header, or empty list if no header found
        
    Example:
    --------
    >>> read_last_header_line('chain.txt')
    ['omega_m', 'sigma_8', 'h', 'n_s', 'log_like', 'weight']
    """
    last_header = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                last_header = line.strip('#').strip()  
            else:
                break  # Stop at first non-comment line

    if last_header:
        return last_header.split() 
    else:
        return []

# =============================================================================
# CONFIGURATION SECTION - Modify these variables for different analyses
# =============================================================================

# List of MCMC chain files to compare
file_paths = ['../sampling/chains/chain_de_binned_shear_lmax3000_mpiparallel_fixcosmo.txt'
 ]  

# Output filename (without extension)
file_name = 'binned_de_cloe_pseudo_fixcosmo_lmax3k'

# Legend labels for each chain (LaTeX formatting supported)
legend_labels = [

]

# Text annotation to display on the plot
#annotation_text = '2500 deg2 \n shear-analysis\n data: HMcode2020 nonlinear\n model: MGEmu'

# Position for annotation (subplot index)
num = 2

# =============================================================================
# DATA LOADING SECTION
# =============================================================================

n_samples = len(file_paths)
chains_info = {}

# Load each MCMC chain file
for i in range(n_samples):
    file_path = file_paths[i]  
    
    # Load numerical data from chain file
    chain = np.genfromtxt(file_path)
    
    # Extract parameter names from header
    chain_pars = read_last_header_line(file_path)
    
    # Remove last two columns (typically log-likelihood and weight)
    chain_pars = chain_pars[:-2]
    
    # Store chain data and parameter names
    chains_info[i] = {}
    chains_info[i]['chain'] = chain
    chains_info[i]['pars'] = chain_pars

# Set parameter ranges for plotting (based on prior ranges)
Ranges = {}
#for par in ['mu0', 'sigma0', 'H0', 'ns', 'logAs', 'Omega_cdm0', 'q1', 'log10TAGN']:
#    Ranges[par] = [fiducials[par]['p1'], fiducials[par]['p2']]

# =============================================================================
# GETDIST SAMPLE PREPARATION
# =============================================================================

samples = []    
for i in range(n_samples):
    # Create GetDist MCSamples object for each chain
    samples.append(
        getdist.MCSamples(
            # Sample data (excluding log-likelihood and weight columns)
            samples = chains_info[i]['chain'][:,:len(chains_info[i]['pars'])],
            # Parameter names
            names = [i for i in chains_info[i]['pars']],
            # Sample weights (exponential of log-likelihood)
            weights=np.exp(chains_info[i]['chain'][:, -2]),
            # LaTeX labels for parameters (from params_dic)
            labels = [params_dic[p] for p in chains_info[i]['pars']],
            # Smoothing settings for 1D and 2D distributions
            settings={'smooth_scale_2D':0.35, 'smooth_scale_1D':0.35},
            # Parameter ranges for plotting
            ranges = Ranges
        ) 
    )
    # Get parameter object (unused but kept for compatibility)
    p = samples[i].getParams() 




# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

# Select parameters to plot (modify as needed)
ModelPars = chains_info[0]['pars']
# Custom parameter selection - take first 5 parameters plus sigma0 and mu0
#ModelPars = ModelPars[:5]+['sigma0', 'mu0'] #, 'q1', 'log10TAGN']

# Set up plotting style and colors
import seaborn as sns
sns.set_theme(style="ticks")
colors = sns.color_palette("Set2")  # Use seaborn Set2 color palette

# Configure LaTeX rendering for publication-quality plots
from matplotlib import rc
rc('text', usetex=True)  # Enable LaTeX text rendering
rc('font',**{'family':'serif','serif':['Times']})  # Use Times font

# Create GetDist subplot plotter
g = getdist.plots.getSubplotPlotter(subplot_size=1.3)

# Configure plot appearance settings
plt.rcParams.update({'font.size':16})
g.settings.legend_fontsize=26      # Legend text size
g.settings.axes_fontsize=18        # Axis tick labels size  
g.settings.axes_labelsize=20       # Axis labels size
g.settings.linewidth=4             # Contour line width
g.settings.figure_legend_frame = False  # Remove legend frame


# =============================================================================
# CREATE TRIANGLE PLOT
# =============================================================================

# Generate the main triangle (corner) plot
g.triangle_plot(
    samples,                          # List of MCSamples objects
    ModelPars,                        # Parameters to include in plot
    legend_labels = legend_labels,    # Labels for legend
    legend_loc = 'upper right',       # Legend position
    # Filled contour styling for each sample
    contour_args = [
        {'filled':True, 'color': colors[0]}, 
        {'filled':True, 'color': colors[1], 'ls': '-'}, 
        {'filled':True, 'color': colors[2], 'ls': '-'},
        {'filled':True, 'color': colors[3], 'ls': '-'},  
        {'filled':True, 'color': colors[4], 'ls': '-'},  
        {'filled':True, 'color': colors[5], 'ls': '-'}
    ], 
    # 1D marginalized distribution styling
    line_args=[
        {'color': colors[0], 'ls': '-'}, 
        {'color': colors[1], 'ls': '-'}, 
        {'color': colors[2], 'ls': '-'}, 
        {'color': colors[3], 'ls': '-'}, 
        {'color': colors[4], 'ls': '-'}, 
        {'color': colors[5], 'ls': '-'}
    ]
)

# =============================================================================
# ADD FIDUCIAL VALUES AND THEORETICAL RELATIONS
# =============================================================================

# Arrays for plotting CAMB priors
mu0_arr = np.linspace(-2., 4., 100)
sigma0_arr = np.linspace(-2., 4., 100)

# Add fiducial values as gray lines if available
if fiducials != None:
    for i in range(len(ModelPars)):
        for j in range(i+1):
            ax = g.subplots[i,j]
            
            # Add horizontal line for fiducial value of y-parameter (for 2D plots)
            if i != j and ModelPars[i] in fiducials and fiducials[ModelPars[i]] != None:
                ax.axhline(fiducials[ModelPars[i]]['p0'], lw=2., color='tab:gray')
            
            # Add vertical line for fiducial value of x-parameter
            if ModelPars[j] in fiducials and fiducials[ModelPars[j]] != None:
                ax.axvline(fiducials[ModelPars[j]]['p0'], lw=2., color='tab:gray')
            
            # Add theoretical relation for specific parameter pair (sigma0 vs mu0)
            #if i==6 and j==5:     
            #    # Numerical condition from CAMB: mu0<=2*sigma0+1
            #    ax.plot(sigma0_arr, 2.*sigma0_arr+1., color='k', ls='--')  

# =============================================================================
# ADD ANNOTATION AND SAVE PLOT
# =============================================================================

# Add text annotation to a specific subplot
#ax = g.subplots[num, num]
#ax.annotate(annotation_text, (1.5, 0.1), xycoords='axes fraction', 
#           clip_on=False, fontsize=20) 

# Save the plot
plt.savefig('figs/posteriors/'+file_name+'.png', dpi=300, bbox_inches='tight')  