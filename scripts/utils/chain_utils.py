#!/usr/bin/env python3
"""
Chain Utilities Module

This module contains utility functions for handling MCMC chain operations,
including header generation and chain formatting for cosmological parameter analysis.

Functions:
    gen_output_header: Generate formatted headers for MCMC chain output files

Author: Maria Tsedrik
Date: 26 Aug 2025
"""


def gen_output_header(header_dic, params_dic):
    """
    Generate a formatted header for MCMC chain output files.
    
    This function creates a comprehensive header containing configuration information,
    observable details, theory model specifications, and parameter priors for 
    cosmological parameter analysis chains.
    
    Parameters:
    -----------
    header_dic : dict
        Dictionary containing analysis configuration information including:
        - observable: Type of observable ('WL', 'GC', '3x2pt', '2x2pt')
        - data: Data file name
        - covariance: Covariance matrix file name  
        - mixmat: Mixing matrix file name
        - nz: Redshift distribution file name
        - lbin: Number of angular frequency bins
        - scale_cuts: Dictionary of scale cuts applied
        - background: Background cosmology model
        - linpert: Linear perturbations model
        - nonlinpert: Nonlinear perturbations model
        
    params_dic : dict
        Dictionary of parameter specifications where each parameter contains:
        - type: Parameter type ('G' for Gaussian, 'U' for Uniform, 'F' for Fixed)
        - p0: Fiducial/fixed value
        - p1: Lower bound (Uniform) or mean (Gaussian)  
        - p2: Upper bound (Uniform) or standard deviation (Gaussian)
    
    Returns:
    --------
    str
        Formatted header string containing all configuration information
        and parameter column names for the chain file
        
    Example:
    --------
    >>> header_dic = {
    ...     'observable': 'WL',
    ...     'data': 'synth_cells.fits',
    ...     'covariance': 'cov_matrix.npy',
    ...     'scale_cuts': {(1,1): [10, 5000]},
    ...     'background': 'camb',
    ...     'linpert': 'hmemu',
    ...     'nonlinpert': 'hmemu'
    ... }
    >>> params_dic = {
    ...     'H0': {'type': 'U', 'p1': 60, 'p2': 80},
    ...     'sigma8': {'type': 'G', 'p1': 0.8, 'p2': 0.05}
    ... }
    >>> header = gen_output_header(header_dic, params_dic)
    """
    
    def get_observable_label(value):
        """
        Convert observable code to descriptive label.
        
        Parameters:
        -----------
        value : str
            Observable code ('WL', 'GC', '3x2pt', '2x2pt')
            
        Returns:
        --------
        str
            Descriptive label for the observable
        """
        observable_map = {
            'WL': "shear-shear",
            'GC': "photo-clustering", 
            '3x2pt': "WL+XC+GC",
            '2x2pt': "XC+GC"
        }
        return observable_map.get(value, "Unknown")
    
    # Get observable description
    observable = get_observable_label(header_dic.get("observable", -1))
    
    # Build main configuration header
    output_header = f"""
        ##############################################################
        CLOE Configuration
        ------------------------------------------------------------
        Observable: {observable}
        Sky Fraction (deg2): 2500
        Data: {header_dic.get("data", None)} 
        Covariance: {header_dic.get("covariance", None)} 
        Mixing matrices: {header_dic.get("mixmat", None)} 
        For 6 bins with n(z): {header_dic.get("nz", None)} 
        Lensing & Clustering Scales:
          - total l_bins: {header_dic.get("lbin", None)} 
          - cut in [l_min, l_max]: 
        """
    
    # Add scale cuts information
    for scalecut_i in header_dic['scale_cuts']:
        output_header += f"\n                     * {scalecut_i}: {header_dic['scale_cuts'][scalecut_i]}"
    
    # Add theory model information
    output_header += f"""

        
        Theory Model:
          - Background: {header_dic.get("background", "N/A")}
          - Linear perturbations: {header_dic.get("linpert", "N/A")}
          - Nonlinear perturbations: {header_dic.get("nonlinpert", "N/A")}
          - Parameter priors: 
        """
    
    # Add parameter prior information
    params_priors = params_dic
    for par_i in params_priors.keys():
        if params_priors[par_i]['type'] == 'G':
            # Gaussian prior: N(mean, std)
            output_header += f"\n           - {par_i}: N({params_priors[par_i]['p1']},{params_priors[par_i]['p2']})"
        elif params_priors[par_i]['type'] == 'U':
            # Uniform prior: [min, max]
            output_header += f"\n           - {par_i}: [{params_priors[par_i]['p1']},{params_priors[par_i]['p2']}]"   
        elif params_priors[par_i]['type'] == 'F':
            # Fixed parameter: value
            output_header += f"\n           - {par_i}: {params_priors[par_i]['p0']}"    
    
    output_header += f"""
        ##############################################################
        """
    
    # Add column headers for the chain data
    # Include only non-fixed parameters in the chain
    params_model = [par for par in params_dic if params_dic[par]['type'] != 'F']
    for par_i in params_model:
        output_header += f"     {par_i}     "
    output_header += "   log_w   log_l"    
    
    return output_header


