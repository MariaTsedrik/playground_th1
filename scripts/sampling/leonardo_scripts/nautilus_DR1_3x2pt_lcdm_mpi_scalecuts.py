# General imports

import os
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from copy import deepcopy
from multiprocessing import get_context
import time
from datetime import timedelta
# MPI parallelization (replaces multiprocessing for HPC clusters)
from mpi4py.futures import MPIPoolExecutor
import yaml
import ast

from nautilus import Prior
from nautilus import Sampler
from scipy.stats import norm

# Import euclidlib for reading the Euclid data
import euclidlib as el

# Import cloelib for cosmology and theoretical predictions
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations

# Import cloelike for likelihoods
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

# We read with euclidlib v2025.2 (v2025.1 is also compatible)


hdf5_name = chain_name = 'checkpoint_3x2pt_DR1_area_1589_covmat_TR1v1.1_lcdm_baryons_chi2totcuts_contaminated'
baryons_in_data = True

data_path = r"/leonardo/home/userexternal/mtsedrik/playground/synthetic_data/synthetic_data_dr1/TR1_13.01.26/"
full_cov=np.load(data_path+'covariance_Gauss_Euclid_DR1_brighter_cut_1589_TR1_2D.npz')['Gauss']


# Get n(z)
z_nz_pos, nz_pos = el.photo.redshift_distributions(data_path+'TR1_v1_Nz_GC_C2020_sel_pv.fits')
z_nz_she, nz_she = el.photo.redshift_distributions(data_path+'TR1_v1_Nz_WL_C2020_sel_pv.fits')


# Normalize and resample n(z) for both position and shear
myz = np.linspace(1e-4, 3.0, 100)
def normalize_and_resample(nz_dict, z_grid, z_target):
    nz_array = np.vstack([nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()])
    return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

my_dndz_pos_norm = normalize_and_resample(nz_pos, z_nz_pos, myz)
my_dndz_she_norm = normalize_and_resample(nz_she, z_nz_she, myz)

fiducial_cosmology = {
    'H0': 67,                # Hubble constant
    'Omega_cdm0': 0.27,        # Cold dark matter density
    'Omega_b0': 0.049,         # Baryon density
    'Omega_k0': 0.0,           # Curvature
    'mnu': 0.077,                # Neutrino mass
    'w0': -1.,                # Dark energy equation of state (present)
    'wa': 0.,                 # Dark energy equation of state (evolution)
    'ns': 0.96,                # Scalar spectral index
    'As': 2.1e-9, # Scalar amplitude
    'gamma_MG': 0.545,         # Modified gravity parameter
    'log10TAGN': 7.75,          # AGN feedback parameter
    'N_mnu': 1,                  # Number of massive neutrino species
    'N_eff': 3.046             # Effective number of relativistic species
}

fiducial_cosmology_nobaryons = {
    'H0': 67,                # Hubble constant
    'Omega_cdm0': 0.27,        # Cold dark matter density
    'Omega_b0': 0.049,         # Baryon density
    'Omega_k0': 0.0,           # Curvature
    'mnu': 0.077,                # Neutrino mass
    'w0': -1.,                # Dark energy equation of state (present)
    'wa': 0.,                 # Dark energy equation of state (evolution)
    'ns': 0.96,                # Scalar spectral index
    'As': 2.1e-9, # Scalar amplitude
    'gamma_MG': 0.545,         # Modified gravity parameter
    'log10TAGN': None,          # AGN feedback parameter
    'N_mnu': 1,                  # Number of massive neutrino species
    'N_eff': 3.046             # Effective number of relativistic species
}

fiducial_nuisance = {
    # Intrinsic alignment
    'AIA': 0.16, 'EtaIA': 1.66, 'CIA': 0.0134,
    # Galaxy bias (poly coefficients)
    'b1_photo_poly0': 1.33291, 'b1_photo_poly1': -0.72414, 
    'b1_photo_poly2': 1.0183, 'b1_photo_poly3': -0.14913,
    # Magnification bias (per bin)
    'magnification_bias_1': 0.0, 'magnification_bias_2': 0.0,
    'magnification_bias_3': 0.0, 'magnification_bias_4': 0.0,
    'magnification_bias_5': 0.0, 'magnification_bias_6': 0.0,
    'magnification_bias_7': 0.0, 'magnification_bias_8': 0.0,
    'magnification_bias_9': 0.0, 'magnification_bias_10': 0.0,
    'magnification_bias_11': 0.0, 'magnification_bias_12': 0.0,
    'magnification_bias_13': 0.0,
    # Redshift errors (positions)
    'dz_pos_1': 0.0, 'dz_pos_2': 0.0, 'dz_pos_3': 0.0, 
    'dz_pos_4': 0.0, 'dz_pos_5': 0.0, 'dz_pos_6': 0.0,
    'dz_pos_7': 0.0, 'dz_pos_8': 0.0, 'dz_pos_9': 0.0, 
    'dz_pos_10': 0.0, 'dz_pos_11': 0.0, 'dz_pos_12': 0.0,
    'dz_pos_13': 0.0,
    # Multiplicative shear bias
    'multiplicative_bias_1': 0.0, 'multiplicative_bias_2': 0.0,
    'multiplicative_bias_3': 0.0, 'multiplicative_bias_4': 0.0,
    'multiplicative_bias_5': 0.0, 'multiplicative_bias_6': 0.0,
    'multiplicative_bias_7': 0.0, 'multiplicative_bias_8': 0.0,
    'multiplicative_bias_9': 0.0, 'multiplicative_bias_10': 0.0,
    'multiplicative_bias_11': 0.0, 'multiplicative_bias_12': 0.0,
    'multiplicative_bias_13': 0.0,
    # Redshift errors (shear)
    'dz_shear_1': 0.0, 'dz_shear_2': 0.0, 'dz_shear_3': 0.0,
    'dz_shear_4': 0.0, 'dz_shear_5': 0.0, 'dz_shear_6': 0.0,
    'dz_shear_7': 0.0, 'dz_shear_8': 0.0, 'dz_shear_9': 0.0, 
    'dz_shear_10': 0.0, 'dz_shear_11': 0.0, 'dz_shear_12': 0.0,
    'dz_shear_13': 0.0
}

ell_theory = np.array([
    10.97557970,
    13.11709027,
    15.67644367,
    18.73516773,
    22.39069761,
    26.75947964,
    31.98068069,
    38.22062129,
    45.67807378,
    54.59059412,
    65.24208925,
    77.97186088,
    93.18541388,
    111.36737359,
    133.09692347,
    159.06625492,
    190.10261691,
    227.19466787,
    271.52396925,
    324.50262398,
    387.81825878,
    463.48778323,
    553.92163814,
    662.00057974,
    791.16744571,
    945.53682628,
    1130.02613377,
    1350.51224608,
    1614.01871364,
    1928.93949355,
    2305.30633773,
    2755.10835283
])

# --- Cosmology and Tracers: Easy to Change! ---

# Cosmology parameters (easy to modify)
cosmo_params = fiducial_cosmology.copy()
nuisance_params = fiducial_nuisance.copy()

# Create background
background = CAMBBackground(
    H0=cosmo_params['H0'],
    Omega_cdm0=cosmo_params['Omega_cdm0'],
    Omega_b0=cosmo_params['Omega_b0'],
    w0=cosmo_params['w0'],
    wa=cosmo_params['wa'],
    Omega_k0=cosmo_params['Omega_k0'],
    ns=cosmo_params['ns'],
    As=cosmo_params['As'],
    mnu=cosmo_params['mnu'],
    gamma_MG=cosmo_params['gamma_MG'],
    N_mnu=cosmo_params['N_mnu']
)

# Choose perturbations
perturbations = HMemuNonLinearPerturbations(
    background, 
    HMemuLinearPerturbations(background, myz), 
    myz, 
    log10TAGN=cosmo_params['log10TAGN'] if baryons_in_data else None
)

# Tracer definitions (easy to swap dndz, bias model, etc.)
tracer_pos = PositionsTracer(
    perturbations=perturbations,
    dndz=my_dndz_pos_norm,
    z=myz,
    galaxy_bias_model='poly',
    nuisance_params=nuisance_params
)

tracer_she = ShearTracer(
    perturbations=perturbations,
    dndz=my_dndz_she_norm,
    z=myz,
    nuisance_params=nuisance_params
)


# 🎯 Compute all the 2-point angular power spectra in one go!
cls_sheshe = AngularTwoPoint(tracer_she, tracer_she).get_Cl(ell_theory, 0, perturbations.k)
cls_posshe = AngularTwoPoint(tracer_pos, tracer_she).get_Cl(ell_theory, 0, perturbations.k)
cls_pospos = AngularTwoPoint(tracer_pos, tracer_pos).get_Cl(ell_theory, 0, perturbations.k)

# 🚀 Combine all spectra into a single dictionary for easy access!
uncoupled_cls = {**cls_posshe, **cls_sheshe, **cls_pospos}

def build_data(cls, mapa_key, cov, zs, nzs, include_pos=False, include_she=False):
    data = {
        'cells': cls,
        'ells': cls[mapa_key].ell,
        'z_arr': zs,
        'cov': cov,
        'mixmat': {},
    }
    if include_pos:
        data['dndz_pos'] = nzs[0]
    if include_she:
        data['dndz_she'] = nzs[1]
    return data

# def build_settings(cls):
#     scale_cuts = {}

#     for key in cls:
#         if key[:2] == ('POS', 'POS'):
#             scale_cuts[key] = [60, 750] #[10, 750]

#         elif key[:2] == ('POS', 'SHE'):
#             scale_cuts[key] = [60, 750] #[10, 750]

#         elif key[:2] == ('SHE', 'SHE'):
#             scale_cuts[key] = [60, 1500] #[10, 1500]

#     return {
#         'n_ell_bins': 32,
#         'scale_cuts': scale_cuts
#     }

def build_settings(cls):
    scales_file = 'scalecuts/scale_cuts_gcggl_linear_shear_chi2tot1p5.yaml'  
    
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

data_3x2pt  = build_data(uncoupled_cls, ('SHE', 'SHE', 1, 1), 
                             full_cov, 
                             myz, [my_dndz_pos_norm, my_dndz_she_norm],
                             include_pos=True, include_she=True)
settings_3x2pt  = build_settings(uncoupled_cls)

like_instance = EuclidLikelihood_3x2pt(
        data=data_3x2pt,
        settings=settings_3x2pt,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="uncoupled"
    )

loglike = like_instance.loglike(fiducial_cosmology | fiducial_nuisance)
print("✨ Log-likelihood at fiducial (3x2pt) with baryons: ", loglike)
print("Fiducial sigma8_0 = ", like_instance.derived['sigma8_0'])

loglike = like_instance.loglike(fiducial_cosmology_nobaryons | fiducial_nuisance)
print("✨ Log-likelihood at fiducial (3x2pt) without baryons: ", loglike)
print("Fiducial sigma8_0 = ", like_instance.derived['sigma8_0'])

# Priors
zmean_dr1 = np.array([0.4371, 0.726, 0.9424, 1.164, 1.437, 1.963])

# Set priors
prior = Prior()
prior.add_parameter('Omega_cdm0', dist=(0.11, 0.7))
prior.add_parameter('Omega_b0', dist=(0.01, 0.1))
prior.add_parameter('H0', dist=(40., 100.))
prior.add_parameter('logAs', dist=(np.log(0.5e-9*1e10), np.log(5e-9*1e10)))
prior.add_parameter('ns', dist=(0.6, 1.2))

#prior.add_parameter('log10TAGN', dist=(7.6, 8.3))
prior.add_parameter('AIA', dist=(-2, 2))
prior.add_parameter('EtaIA', dist=(-5, 5))
#prior.add_parameter('w0', dist=(-3.0, -0.3))
#prior.add_parameter('wa', dist=(-3.0, 3.0))
prior.add_parameter('b1_photo_poly0', dist=(-3.0, 3.0))
prior.add_parameter('b1_photo_poly1', dist=(-3.0, 3.0))
prior.add_parameter('b1_photo_poly2', dist=(-3.0, 3.0))
prior.add_parameter('b1_photo_poly3', dist=(-3.0, 3.0))
prior.add_parameter('multiplicative_bias_1', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_2', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_3', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_4', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_5', norm(loc=0.0, scale=0.01))
prior.add_parameter('multiplicative_bias_6', norm(loc=0.0, scale=0.01)) 
# prior.add_parameter('magnification_bias_1', dist=(-2.0, 2.0))
# prior.add_parameter('magnification_bias_2', dist=(-2.0, 2.0))
# prior.add_parameter('magnification_bias_3', dist=(-2.0, 2.0))
# prior.add_parameter('magnification_bias_4', dist=(-2.0, 2.0))
# prior.add_parameter('magnification_bias_5', dist=(-2.0, 2.0))
# prior.add_parameter('magnification_bias_6', dist=(-2.0, 2.0))
prior.add_parameter('dz_pos_1', norm(loc=0.0, scale=0.002*(1+zmean_dr1[0])))
prior.add_parameter('dz_pos_2', norm(loc=0.0, scale=0.002*(1+zmean_dr1[1])))
prior.add_parameter('dz_pos_3', norm(loc=0.0, scale=0.002*(1+zmean_dr1[2])))
prior.add_parameter('dz_pos_4', norm(loc=0.0, scale=0.002*(1+zmean_dr1[3])))
prior.add_parameter('dz_pos_5', norm(loc=0.0, scale=0.002*(1+zmean_dr1[4])))
prior.add_parameter('dz_pos_6', norm(loc=0.0, scale=0.002*(1+zmean_dr1[5])))
prior.add_parameter('dz_shear_1', norm(loc=0.0, scale=0.002*(1+zmean_dr1[0])))
prior.add_parameter('dz_shear_2', norm(loc=0.0, scale=0.002*(1+zmean_dr1[1])))
prior.add_parameter('dz_shear_3', norm(loc=0.0, scale=0.002*(1+zmean_dr1[2])))
prior.add_parameter('dz_shear_4', norm(loc=0.0, scale=0.002*(1+zmean_dr1[3])))
prior.add_parameter('dz_shear_5', norm(loc=0.0, scale=0.002*(1+zmean_dr1[4])))
prior.add_parameter('dz_shear_6', norm(loc=0.0, scale=0.002*(1+zmean_dr1[5])))

print("✨ Prior set up completed.")
print("Parameters sampled: ", prior.keys)
print("Number of sampled parameters:", len(prior.keys))

default_pars = fiducial_cosmology_nobaryons | fiducial_nuisance

def like_Naut_test(param_dict):
    # Enforce physical prior on w0 + wa
    #if param_dict['w0'] + param_dict['wa'] >= 0:
    #    return (-np.inf, 0)

    pars_in = default_pars.copy()
    pars_in.update(param_dict)

    h = pars_in['H0'] / 100.0
    # need these parameters for cloe
    # uncomment for cmb-priors!!!
    #pars_in['Omega_cdm0'] = param_dict['omch2'] / h**2
    #pars_in['Omega_b0'] = param_dict['ombh2'] / h**2
    #pars_in['As'] = np.exp(param_dict['logAs']) * 1e-10
    try:
        like = like_instance.loglike(pars_in) 
    except ValueError:
        like = -np.inf

    return (like, like_instance.derived['sigma8_0'])

blob_vec     = [('sigma8_0', float)]

parallelisation_option = "MPI"
sampling_workers = 32

if __name__ == '__main__':
    # Initialize Nautilus sampler 
    sampler = Sampler(
        prior, 
        like_Naut_test, 
        n_live=5000,
        filepath='chains/hdf5/'+hdf5_name+'.hdf5', 
        resume=False, 
        pool=MPIPoolExecutor()  # MPI-based parallelization
    )
    start = time.time()
    sampler.run(verbose=True, discard_exploration=True)
    finish = time.time()
    chain_time = finish - start

    # Extract results
    log_z = sampler.log_z
    #points, log_w, log_l = sampler.posterior()
    points, log_w, log_l, derived = sampler.posterior(return_blobs=True)


    # Save text chain with header and timing footer
    #output_header = gen_output_header(header_dic, params_dic)
    chain_time_hms = timedelta(seconds=chain_time)
    footer = f'log_Z = {log_z}; chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'

    # --- Sampled parameters ---
    param_names = list(prior.keys)
    chain_dict = {
        name: points[:, i]
        for i, name in enumerate(param_names)
    }

    print('derived: ', derived)

    # --- Derived parameters ---
    derived_names = [name for name, _ in blob_vec]
    print('derived names: ', derived_names)

    derived_dict = {
        derived_names[0]: np.array(derived)
    }
    
    # --- Save everything in a labelled way ---
    np.savez_compressed(
        'chains/chain_' + chain_name + '.npz',
        chain=chain_dict,
        derived=derived_dict,
        weights=log_w,
        logl=log_l,
        param_names=param_names,
        derived_names=derived_names,
        logz=log_z,
        chain_time=chain_time,
        chain_time_hms=chain_time_hms,
        fiducial=default_pars
    )


    print(f"{parallelisation_option} with {sampling_workers} processes: sampling complete!")
    print(f"  Evidence: log_Z = {log_z:.6f}")
    print(f"  Total time: {chain_time:.2f} s ({chain_time_hms})")
    print(f"  Output files:")
    print(f"    HDF5: chains/hdf5/{hdf5_name}.hdf5")
    print(f"    Chain: chains/chain_{chain_name}.npz")

