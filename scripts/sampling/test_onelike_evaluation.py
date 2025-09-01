#!/usr/bin/env python3
"""
Cosmological Likelihood Evaluation Test Script

This script tests the performance of cosmological likelihood evaluations using
various theory models (ΛCDM, Modified Gravity) for Euclid survey data.
It measures timing for likelihood computations and generates chain headers.

Key Features:
- Thread control for HPC cluster environments (Cuillin, Edinburgh)
- Support for multiple cosmology models (ΛCDM, Modified Gravity)
- Likelihood timing benchmarks with JIT compilation handling
- Chain header generation for MCMC analysis

Threading Control:
The script sets strict threading limits before importing TensorFlow to prevent
thread oversubscription on HPC clusters. This is critical because:
- HMcode2020Emu eagerly loads TensorFlow-based cosmopower emulator
- cosmopower creates persistent thread pools that can cause performance issues
- Without proper threading control, evaluations can be 10x slower

Supported Models:
- LCDM_NL: Standard ΛCDM with nonlinear corrections
- MG_L: Modified Gravity with linear perturbations only  
- MG_NL: Modified Gravity with nonlinear corrections

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

# CLOE likelihood modules  
from cloelike.EuclidLikelihood_3x2pt_Cls import EuclidLikelihood_3x2pt_Cls
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
    scales_file = 'scalecuts/scale_cuts.yaml'
    
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
    if type_name == 'LCDM_NL':
        # Standard ΛCDM model with HMcode2020 emulator
        like_instance = EuclidLikelihood_WL_Cls(
            data=data_WL,
            settings=settings_WL,
            Background=CAMBBackground,
            LinPerturbations=HMemuLinearPerturbations,
            NonLinPerturbations=HMemuNonLinearPerturbations,
        )
        linpert_name = 'hmemu'
        nonlinpert_name = 'hmemu'
        
    elif type_name == 'MG_L':
        # Modified Gravity with linear perturbations only
        like_instance = EuclidLikelihood_WL_Cls(
            data=data_WL,
            settings=settings_WL,
            Background=CAMBBackground,
            LinPerturbations=MGrowthLinearPerturbations,
            LinPerturbationsBase=HMemuLinearPerturbations,
            NonLinPerturbations=None,
            gravity_model='musigma-de'
        )
        linpert_name = 'hmemu+mgrowth'
        nonlinpert_name = 'none'
        
    elif type_name == 'MG_NL':   
        # Modified Gravity with nonlinear boost corrections
        like_instance = EuclidLikelihood_WL_Cls(
            data=data_WL,
            settings=settings_WL,
            Background=CAMBBackground,
            LinPerturbations=MGrowthLinearPerturbations, 
            LinPerturbationsBase=HMemuLinearPerturbations,
            NonLinPerturbations=BoostedPerturbations,
            gravity_model='musigma-de'
        ) 
        linpert_name = 'hmemu+mgemu'
        nonlinpert_name = 'hmemu+mgemu'
        
    else:
        raise ValueError(f"type_name '{type_name}' not allowed. "
                        f"Must be one of: 'LCDM_NL', 'MG_L', 'MG_NL'")
    
    # Build header dictionary for chain output
    header_dic = {
        'observable': 'WL',
        'data': data_file,
        'covariance': cov_file,
        'mixmat': mixmat_file,
        'nz': nz_file,
        'lbin': settings_WL['n_ell_bins'],
        'scale_cuts': settings_WL['scale_cuts'],
        'background': 'camb',
        'linpert': linpert_name,
        'nonlinpert': nonlinpert_name,
    }
    
    return like_instance, header_dic

# =============================================================================
# ANALYSIS CONFIGURATION AND PARAMETER SETUP
# =============================================================================

# Select cosmological model for analysis
type_name = 'MG_L'  # Options: 'LCDM_NL', 'MG_L', 'MG_NL'
like_instance, header_dic = get_like_instance(type_name)

# Load parameter configuration
config_file = 'inifiles/params_model_shear.yaml'
with open(config_file, "r") as file_in:
    params_dic = yaml.safe_load(file_in)

# Extract parameter categories
params_priors = params_dic
params_model = [par for par in params_dic if params_dic[par]['type'] != 'F']  # Varying parameters
params_fixed = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] == 'F'}  # Fixed parameters

# =============================================================================
# LIKELIHOOD EVALUATION TESTING
# =============================================================================

# Set up test parameter values (fiducial values)
chain_name = 'test_header_newlike_mgrowth'
dic_test = {par_i: params_priors[par_i]['p0'] for par_i in params_priors.keys()}
dic_test = params_fixed | dic_test  # Merge fixed and varying parameters
# Prepare test parameter dictionary with derived parameters
# Convert from h-based to physical parameters as required by likelihood
dic_test['Omega_b0'] = dic_test['ombh2'] / (dic_test['H0']/100)**2
dic_test['As'] = np.exp(dic_test['logAs'])*1e-10

# Perform multiple likelihood evaluations to test JIT compilation effects
# First call includes JIT compilation overhead, subsequent calls are optimized

print("Testing likelihood evaluation performance...")

# First evaluation (includes JIT compilation)
start = time.time()
test_like = like_instance.loglike(dic_test)   
finish = time.time()
test_time = finish - start
test_time_hms = timedelta(seconds=test_time)        

test_txt = f"""
##############################################################
# First call (includes JIT compilation)
# loglikelihood = {test_like:.6f}
# evaluation took {test_time:.4f} s (--> {test_time_hms} hh:mm:ss)
"""

# Second evaluation (JIT compiled, should be faster)
start = time.time()
test_like = like_instance.loglike(dic_test)   
finish = time.time()
test_time = finish - start
test_time_hms = timedelta(seconds=test_time)        

test_txt += f"""
##############################################################
# Second call (JIT compiled)
# loglikelihood = {test_like:.6f}
# evaluation took {test_time:.4f} s (--> {test_time_hms} hh:mm:ss)
"""

# Third evaluation (confirm consistent timing)
start = time.time()
test_like = like_instance.loglike(dic_test)   
finish = time.time()
test_time = finish - start
test_time_hms = timedelta(seconds=test_time)        

test_txt += f"""
##############################################################
# Third call (timing verification)
# loglikelihood = {test_like:.6f}
# evaluation took {test_time:.4f} s (--> {test_time_hms} hh:mm:ss)
"""

print(f"Likelihood evaluation complete. Final value: {test_like:.6f}")

# Save chain header with timing information
output_header = gen_output_header(header_dic, params_dic) + test_txt
np.savetxt("chains/chain_"+chain_name+".txt", [], header=output_header)

print(f"Chain header saved to: chains/chain_{chain_name}.txt")