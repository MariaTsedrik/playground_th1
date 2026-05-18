#!/usr/bin/env python3
"""
Nautilus Sampling Example for Cosmological Parameter Inference

This script demonstrates the use of the Nautilus sampler for cosmological
parameter inference using weak lensing data from Euclid survey.
It implements nested sampling with parallel likelihood evaluation for
efficient exploration of high-dimensional parameter spaces.

Key Features:
- Thread control for HPC cluster environments (Cuillin, Edinburgh)
- Support for multiple cosmology models (ΛCDM, Modified Gravity)
- Nested sampling with Nautilus for evidence computation
- Parallel likelihood evaluation with multiprocessing
- Chain output with comprehensive headers and timing information

Threading Control:
The script sets strict threading limits before importing TensorFlow to prevent
thread oversubscription on HPC clusters. This is critical because:
- HMcode2020Emu loads TensorFlow-based cosmopower emulator
- cosmopower creates persistent thread pools that can cause performance issues
- Without proper threading control, evaluations can be 10x slower

Supported Models:
- LCDM_NL: Standard ΛCDM with nonlinear corrections
- MG_L: Modified Gravity with linear perturbations only  
- MG_NL: Modified Gravity with nonlinear corrections

Sampling Configuration:
- Uses Nautilus nested sampler with 3000 live points
- Targets effective sample size of 5000
- Parallel processing with 4 workers for sampling
- Outputs both HDF5 and text chain formats

Author: Maria Tsedrik
Date: 26 Aug 2025
"""

# =============================================================================
# CRITICAL: Thread Control for HPC Clusters
# =============================================================================
# Must be set BEFORE importing TensorFlow to prevent thread oversubscription
# HMcode2020Emu loads TensorFlow-based cosmopower emulator during
# instantiation, creating persistent thread pools across the entire process

import os
# Set environment variables to control threading
os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL threads  
os.environ["TF_NUM_INTEROP_THREADS"] = "1"   # TensorFlow inter-op threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"   # TensorFlow intra-op threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"     # OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # NumExpr threads
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable oneDNN optimizations

# Configure TensorFlow threading after environment setup
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
_ = tf.constant(0)  # Dummy operation to trigger TensorFlow initialization

# Verify threading configuration
print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())

# Suppress TensorFlow and warning messages for cleaner output
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from copy import deepcopy
import time
from datetime import timedelta
import yaml
import ast
import multiprocessing
import sys

# =============================================================================  
# EXTERNAL LIBRARY IMPORTS
# =============================================================================

# Euclid data handling
import euclidlib as el

# CLOE cosmology and theory modules
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
from cloelib.cosmology.mgrowth_cosmology import MGrowthLinearPerturbations
from cloelib.cosmology.reactemu_cosmology import MGemuNonlinearBoost, BoostedPerturbations
from cloelib.cosmology.wz_cosmology import DEBackground, DELinearPerturbations, DENonlinearPerturbations

# CLOE likelihood modules  
from cloelike.EuclidLikelihood_WL_Cls import EuclidLikelihood_WL_Cls

# Sampling and statistical tools
from nautilus import Prior
from nautilus import Sampler
from scipy.stats import norm

# Local utilities
sys.path.append('../utils')
from chain_utils import gen_output_header


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def normalize_and_resample(nz_dict, z_grid, z_target):
    """
    Normalize and resample redshift distributions.
    
    Takes a dictionary of redshift distributions, normalizes each to unit area,
    and resamples onto a target redshift grid using linear interpolation.
    
    Parameters:
    -----------
    nz_dict : dict
        Dictionary containing redshift distributions for each bin
    z_grid : array_like
        Original redshift grid
    z_target : array_like  
        Target redshift grid for resampling
        
    Returns:
    --------
    numpy.ndarray
        Array of normalized and resampled distributions, shape (n_bins, len(z_target))
    """
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

# Load redshift distributions from FITS file
nz_file = 'nz_example.fits'
z_nz, nz_heracles = el.photo.redshift_distributions('../../tutorials/th1-kp4/'+nz_file)

# Define target redshift grid for analysis
myz = np.linspace(1e-4, 3.0, 100)

# Normalize and resample n(z) for position and shear samples
my_dndz_pos_norm = normalize_and_resample(nz_heracles, z_nz, myz)
my_dndz_she_norm = normalize_and_resample(nz_heracles, z_nz, myz)

# Load angular power spectra data (compatible with euclidlib v2025.1+)
data_file = 'synth_cells_5000_binned.fits'
cells_data = el.photo.angular_power_spectra('../../tutorials/th1-kp4/'+data_file)

# Load mixing matrices for systematic corrections
mixmat_file = 'mixmat_identity_5000_binned.fits'
mixmat = el.photo.mixing_matrices('../../tutorials/th1-kp4/'+mixmat_file)

# Load covariance matrix (follows seaborne format for full 3x2pt analysis)
cov_file = 'cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode_copy.npy'
full_cov = np.load('../../tutorials/th1-kp4/'+cov_file)

# Extract weak lensing covariance submatrix
# Shape: (n_tomographic_pairs * n_ell_bins, n_tomographic_pairs * n_ell_bins)
# For 6 bins: (21+21)*32 = 1344 elements (EE + BB correlations)
cov_WL = full_cov[:(21+21)*32, :(21+21)*32]


def build_data(ell_key, cov, include_pos=False, include_she=False):
    """
    Build data dictionary for likelihood analysis.
    
    Constructs a data dictionary containing all necessary components for
    cosmological likelihood evaluation, including power spectra, covariance,
    and redshift distributions.
    
    Parameters:
    -----------
    ell_key : tuple
        Key specifying which power spectrum type (e.g., ('SHE', 'SHE', 1, 1) for shear)
    cov : numpy.ndarray
        Covariance matrix for the specified observable
    include_pos : bool, optional
        Whether to include position (clustering) redshift distributions
    include_she : bool, optional
        Whether to include shear (weak lensing) redshift distributions
        
    Returns:
    --------
    dict
        Data dictionary with keys: 'cells', 'ells', 'z_arr', 'cov', 'mixmat',
        and optionally 'dndz_pos', 'dndz_she'
    """
    data = {
        'cells': cells_data,
        'ells': cells_data[ell_key].ell,
        'z_arr': myz,
        'cov': cov,
        'mixmat': mixmat,
    }
    if include_pos:
        data['dndz_pos'] = my_dndz_pos_norm
    if include_she:
        data['dndz_she'] = my_dndz_she_norm
    return data

def build_settings():
    """
    Build analysis settings from configuration files.
    
    Reads scale cuts from YAML configuration and converts string keys
    to tuple format for tomographic bin pairs.
    
    Returns:
    --------
    dict
        Settings dictionary containing 'n_ell_bins' and 'scale_cuts'
    """
    scales_file = 'scalecuts/scale_cuts_3000.yaml'  # Scale cuts for kmax = 0.3 h/Mpc
    
    # Load scale cuts configuration
    with open(scales_file, "r") as file_in:
        scale_cuts = yaml.safe_load(file_in)
    
    # Convert string keys to tuples for tomographic bin pairs
    # e.g., "(1,1)" -> (1,1) for proper dictionary indexing
    converted = {}
    for k, v in scale_cuts.items():
        try:
            tuple_key = ast.literal_eval(k)
            if isinstance(tuple_key, tuple):
                converted[tuple_key] = v
            else:
                converted[k] = v
        except (SyntaxError, ValueError):
            converted[k] = v
    
    return {'n_ell_bins': 32, 'scale_cuts': converted}

# =============================================================================
# DATASET CONSTRUCTION
# =============================================================================

# Build weak lensing dataset with shear redshift distributions
data_WL = build_data(('SHE', 'SHE', 1, 1), cov_WL, include_she=True)

# Build analysis settings (scale cuts, binning configuration)
settings_WL = build_settings()

# =============================================================================
# LIKELIHOOD CONSTRUCTION
# =============================================================================

def get_like_instance(type_name):
    """
    Create likelihood instance for different cosmological models.
    
    Factory function that creates appropriate likelihood objects for different
    combinations of background cosmology, linear perturbations, and nonlinear
    corrections.
    
    Parameters:
    -----------
    type_name : str
        Model type identifier:
        - 'LCDM_NL': Standard ΛCDM with HMcode2020 nonlinear corrections
        - 'MG_L': Modified Gravity with linear perturbations only
        - 'MG_NL': Modified Gravity with nonlinear boost corrections
        
    Returns:
    --------
    tuple
        (likelihood_instance, header_dictionary) where header_dictionary
        contains configuration information for chain output
        
    Raises:
    -------
    ValueError
        If type_name is not one of the supported model types
    """
    zfinal = 1000.
    zmax = 3.
    zmin = 0.
    zbin_edges = np.array([0., 0.4, 0.8, 1.2, 1.6, 2., zmax, zfinal])
    zbin_centers = 0.5*(zbin_edges[1:] + zbin_edges[:-1])
    zbin_widths = np.diff(zbin_edges)
    
    Nbin = len(zbin_centers)

    print('z-bin edges: ', zbin_edges)
    print('z-bin centers: ', zbin_centers)
    print('z-bin widths: ', zbin_widths)

    if type_name == 'de_bin':
        # Modified Gravity with nonlinear boost corrections
        like_instance = EuclidLikelihood_WL_Cls(
            data=data_WL,
            settings=settings_WL,
            Background=DEBackground,
            BackgroundLCDM=CAMBBackground,
            LinPerturbations=DELinearPerturbations, 
            LinPerturbationsBase=HMemuLinearPerturbations,
            NonLinPerturbations=DENonlinearPerturbations
        ) 
        linpert_name = 'hmemu+mgrowth'
        nonlinpert_name = 'pseudo-hmemu'
        
    else:
        raise ValueError(f"type_name '{type_name}' not allowed. ")
    
    # Build header dictionary for chain output
    header_dic = {
        'observable': 'WL',
        'data': data_file,
        'covariance': cov_file,
        'mixmat': mixmat_file,
        'nz': nz_file,
        'lbin': settings_WL['n_ell_bins'],
        'scale_cuts': settings_WL['scale_cuts'],
        'background': 'de',
        'linpert': linpert_name,
        'nonlinpert': nonlinpert_name,
    }
    
    return like_instance, header_dic
# ⚠️  IMPORTANT: Verify scale cuts file matches your analysis requirements
# Current file: scale_cuts_0p3.yaml (for kmax = 0.3 h/Mpc)
hdf5_name = chain_name = 'debinned_shear_lmax3000_nonparallel_fixcosmo'
type_name = 'de_bin' #'LCDM_NL', 'MG_L' or 'MG_NL
like_instance, header_dic = get_like_instance(type_name)
config_file = 'inifiles/params_model_shear_de.yaml'
###################
with open(config_file, "r") as file_in:
    params_dic = yaml.safe_load(file_in)
params_priors = params_dic
params_model = [par for par in params_dic if params_dic[par]['type'] != 'F']
params_fixed = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] == 'F'}

print(gen_output_header(header_dic, params_dic))

# =============================================================================
# SAMPLING SETUP AND PRIOR DEFINITION
# =============================================================================

# Build prior distribution for varying parameters
prior = Prior()
for par_i in params_model:
    if params_priors[par_i]['type'] == 'G':
        # Gaussian prior: N(mean, std)
        prior.add_parameter(par_i, dist=norm(loc=params_priors[par_i]['p1'], scale=params_priors[par_i]['p2']))
    elif params_priors[par_i]['type'] == 'U':
        # Uniform prior: [min, max]
        prior.add_parameter(par_i, dist=(params_priors[par_i]['p1'], params_priors[par_i]['p2']))


def like_Naut_test(param_dict):
    """
    Likelihood function for Nautilus sampler.
    
    Converts input parameters to the format expected by CLOE likelihood,
    handles parameter transformations, and catches evaluation errors.
    
    Parameters:
    -----------
    param_dict : dict
        Dictionary of varying parameters from the sampler
        
    Returns:
    --------
    float
        Log-likelihood value, or -inf if evaluation fails
        
    Notes:
    ------
    - Converts h-based parameters to physical units (Omega_b0, As)
    - Merges varying and fixed parameters
    - Returns -inf for invalid parameter combinations
    """
    # Merge varying and fixed parameters
    pars_in = param_dict | params_fixed
    
    # Convert to CLOE base parameters (h-based to physical)
    pars_in['Omega_b0'] = pars_in['ombh2'] / (pars_in['H0']/100)**2
    pars_in['As'] = np.exp(pars_in['logAs'])*1e-10
    
    try:
        like = like_instance.loglike(pars_in)
    except ValueError:
        # Return -inf for invalid parameter combinations
        like = -np.inf
    
    return like


params_model = [par for par in params_dic if params_dic[par]['type'] != 'F']
params_fixed = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] == 'F'}
       
# =============================================================================
# LIKELIHOOD TESTING
# =============================================================================

# Test likelihood evaluation with fiducial parameters
dic_test = {par_i: params_priors[par_i]['p0'] for par_i in params_priors.keys()}
dic_test = params_fixed | dic_test

print('Test likelihood evaluation:')
print(f'  Fiducial parameters: {dic_test}')
print(f'  Log-likelihood: {like_Naut_test(dic_test):.6f}')

# =============================================================================
# SAMPLING EXECUTION
# =============================================================================

# ⚠️  IMPORTANT: Multiprocessing considerations
# 
# From Nautilus documentation: to parallelize likelihood evaluations but 
# not sampler calculations, use pool=(4, None).
# Pool configuration: pool=(n_likelihood_workers, n_sampling_workers)
# - n_sampling_workers=4: Parallel processes for sampling coordination
# - n_likelihood_workers=1: Single likelihood evaluation
#
# WARNING: os.fork() is incompatible with multithreaded JAX code
# This can lead to deadlocks. Consider using MPI for production runs.
# 
# For development/testing, the current setup works but monitor for issues.

sampling_workers = 1
def main():    
    # Initialize Nautilus sampler
    sampler = Sampler(
        prior, 
        like_Naut_test, 
        n_live=3000,
        filepath='chains/hdf5/'+hdf5_name+'.hdf5', 
        resume=False, 
        pool=(1, sampling_workers)  # (likelihood_workers, sampling_workers)
    )
    
    # Run sampling with timing
    start = time.time()
    sampler.run(verbose=True, discard_exploration=True, n_eff=5000)
    
    # Extract results
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = finish - start

    # Save text chain with header and timing footer
    output_header = gen_output_header(header_dic, params_dic)
    chain_time_hms = timedelta(seconds=chain_time)
    footer = f'log_Z = {log_z}; chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'
    
    np.savetxt(
        "chains/chain_"+chain_name+".txt", 
        np.c_[points, log_w, log_l], 
        header=output_header, 
        footer=footer
    )
    
    print(f"Sampling complete!")
    print(f"  Evidence: log_Z = {log_z:.6f}")
    print(f"  Total time: {chain_time:.2f} s ({chain_time_hms})")
    print(f"  Output files:")
    print(f"    HDF5: chains/hdf5/{hdf5_name}.hdf5")
    print(f"    Chain: chains/chain_{chain_name}.txt")



if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all multiprocessing pools are properly closed
        # This prevents resource leaks and hanging processes
        multiprocessing.active_children()
