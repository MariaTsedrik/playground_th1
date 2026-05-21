"""
Nautilus Sampler with cloelike using synthetic spectro data
==========================================================
This script demonstrates how to:
1. Set up a likelihood function from cloelike
2. Define a prior distribution with nautilus
3. Connect them through a wrapper function
4. Run Bayesian inference with the Nautilus sampler
"""

# ============================================================================
# IMPORTS
# ============================================================================

# General imports
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from itertools import product
from copy import deepcopy
from dataclasses import replace
import time
from nautilus import Prior, Sampler
from scipy.stats import norm
import euclidlib
import multiprocessing

# cloelib imports
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations

# cloelike imports
from cloelike.EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls

# Avoid warnings for cleaner output 
np.seterr(divide='ignore', over='ignore', invalid='ignore') # Ignore divide by zero and overflow errors in numpy operations, which can occur in loglike calculation, it is annoying otherwise


# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

labels_file = ['085-110','110-145','145-185']

folder_path = '../../tutorials/th1-kp4/GC_TR1_2ndit/'

# This uses the functionality of euclidlib of reading multiple files at the same time
filename = 'mps_pk_{}_synth.fits'
datavec = euclidlib.le3.pk_gc.power_spectrum_multipoles(folder_path+filename, *labels_file)

filename = 'Analytical_Cov_EWS_DR1_rescaled_TR1_iter2_{}_Tot_rebinned_wNP0.fits'
covariance = euclidlib.le3.pk_gc.power_spectrum_multipole_covariance(folder_path+filename, *labels_file)

filename = 'mixing_matrix_EWS_TR1_iter2_{}_Tot_rebinned.fits'
mixing = euclidlib.le3.pk_gc.power_spectrum_multipole_mixing_matrix(folder_path+filename, *labels_file)



# Redshifts of the samples (we transform them to strings to have identifiers of the bins)
redshifts = np.array([0.97, 1.27, 1.63])
labels = [str(z).strip('0') for z in np.round(redshifts,2)]
# The data dictionary is what must be passed as input to the likelihood class
data = {
    # Data (we fill this part below)
    'GCspectro': {},
    # Fiducial cosmology (required for geometrical (AP) distortions)
    'fiducial_cosmology': {}
}

# Storing fiducial cosmology

# Storing fiducial cosmology
dv = datavec[("SPE", "SPE", 0, 0)]
data['fiducial_cosmology']['H0'] = 67.0
data['fiducial_cosmology']['Omega_cdm0'] = 0.27
data['fiducial_cosmology']['Omega_b0'] = 0.049
data['fiducial_cosmology']['Omega_k0'] = 0
data['fiducial_cosmology']['mnu'] = 0.0
data['fiducial_cosmology']['N_mnu'] = 0
data['fiducial_cosmology']['w0'] = -1
data['fiducial_cosmology']['wa'] = 0.0
data['fiducial_cosmology']['ns'] = 0.96
data['fiducial_cosmology']['As'] = 2.1e-9
data['fiducial_cosmology']['gamma_MG'] = 0.545

# Rescaling of the units (as, at the moment, cloelib works in Mpc units, while the data vectors are stored in Mpc/h units)
fid_h = data['fiducial_cosmology']['H0'] / 100.0
k_fac = fid_h
pk_fac = 1.0 / fid_h**3
cov_fac = 1.0 / fid_h**6


for ii,z in enumerate(labels):
    # Looping over the identifiers (redshifts)
    data['GCspectro'][z] = {}
    dv_inst = datavec[('SPE', 'SPE', ii, ii)]
    cv_inst = covariance[('SPE', 'SPE', ii, ii)]
    mm_inst = mixing[('SPE', 'SPE', ii, ii)]

    # Storing number densities
    data['GCspectro'][z]['nbar'] = dv_inst.nbar

    kvec_mask = mm_inst.kout <= dv_inst.keff[-1]
    # Storing data vectors
    data['GCspectro'][z]['k'] = dv_inst.keff * k_fac
    data['GCspectro'][z]['pk0'] = dv_inst.multipoles[0] * pk_fac
    data['GCspectro'][z]['pk2'] = dv_inst.multipoles[2] * pk_fac
    data['GCspectro'][z]['pk4'] = dv_inst.multipoles[4] * pk_fac

    # Storing covariance matrices
    full_matrix = np.block([[cv_inst.covariance[f'ELL_{i}-{j}'] for j in [0, 2, 4]] for i in [0, 2, 4]])
    data['GCspectro'][z]['cov'] = full_matrix * cov_fac

    # Storing mixing matrices
    resc_kout = mm_inst.kout * k_fac
    resc_kin = {ell: val * k_fac for ell, val in mm_inst.kin.items()}
    squeezed_mixing = {key: val.squeeze() for key, val in mm_inst.mixing.items()}
    
    replace(
        mm_inst, kout=mm_inst.kout * k_fac)
    data['GCspectro'][z]['mixing_matrix'] = replace(
        mm_inst, kout=resc_kout, kin=resc_kin, mixing=squeezed_mixing
    )

# ============================================================================
# 2. BUILD LIKELIHOOD DATA AND SETTINGS
# ============================================================================

settings = {
    'scale_cuts': {
        'GCspectro': {
            'bin1': {'ell0': [0.0, 0.20], 'ell2': [0.0, 0.20], 'ell4': [0.0, 0.20]},
            'bin2': {'ell0': [0.0, 0.20], 'ell2': [0.0, 0.20], 'ell4': [0.0, 0.20]},
            'bin3': {'ell0': [0.0, 0.20], 'ell2': [0.0, 0.20], 'ell4': [0.0, 0.20]},
        }
    }
}

# Also in this case we have passed the scale cuts in Mpc/h units, so we need to transform them into Mpc
settings = {
    'scale_cuts': {
        'GCspectro': {
            bin_key: {ell_key: [v * fid_h for v in values] for ell_key, values in bin_values.items()}
            for bin_key, bin_values in settings['scale_cuts']['GCspectro'].items()
        }
    }
}

# ============================================================================
# 3. INITIALIZE LIKELIHOOD OBJECT
# ============================================================================

# Initialize likelihood instance with cosmology and perturbation modules
from cloelib.observables.PBJ_spectro import PBJSpectroPower

print("🔍 Timing log-likelihood initialisation for spectroscopic GC:")

start = time.time()

like_spec_pbj = EuclidLikelihood_GCspectro_Pls(
    data=data,
    settings=settings,
    Background=CAMBBackground,
    SpectroPower=PBJSpectroPower,
    Perturbations=HMemuLinearPerturbations
)

end = time.time()

print(f"✅ Spectroscopic GC likelihood initialised.")
print(f"⏱️ Time elapsed: {end - start:.3f} seconds")

# These are the fiducial parametes for COMET
parameters = {
    'H0': 67.0,
    'Omega_cdm0': 0.27, 'Omega_b0': 0.049, 'Omega_k0': 0.0, 'mnu': 0.00001, 'N_mnu': 1,
    'w0': -1.0, 'wa': 0.0,
    'ns': 0.96, 'As': 2.1e-9,
    'gamma_MG': 0.545,
    'b1': np.array([1.537, 2.07, 2.1]),
    'b2': np.array([0.695, 0.870, 1.162]),
    'bG2': np.array([-0.156, -0.299, -0.400]),
    'bGam3':  np.array([3.46121933e+00,  1.84293956e+00,  8.35302819e+00]),
    'c0':     np.array([4.92649455e+01,  2.28405327e+01, -1.61265049e+00]),
    'c2':     np.array([ 4.31120206e+01,  3.18420310e+01,  6.74422117e+01]),
    'c4':     np.array([-1.25077281e+01, -2.84513988e+01, -6.76616588e+01]),
    'cnlo':   np.array([-4.34672822e+02, -1.08102331e+03, -2.06601882e+02]),
    'NP0':    np.array([1.30543441e+00,  2.26616744e-01,  9.12768142e-01]),
    'NP20':   np.array([-5.62410948e+00, -3.73494300e+00, -1.22546846e+01]),
    'NP22':   np.array([-5.94837043e-01, -5.18488492e+00,  2.96041321e+00]),
    'fout':   np.array([0.5, 0.5, 0.5]),
    'sigmaz': np.array([0.0, 0.0, 0.0])}



## Calculate fiducial f to convert COMET pars into PBJ pars
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations

background = CAMBBackground(
    H0=parameters['H0'], Omega_cdm0=parameters['Omega_cdm0'], Omega_b0=parameters['Omega_b0'], Omega_k0=parameters['Omega_k0'],
    w0=parameters['w0'], wa=parameters['wa'], ns=parameters['ns'], As=parameters['As'], gamma_MG=parameters['gamma_MG'], mnu=0.0001, N_mnu=1
)

linear_perturbations_emu = HMemuLinearPerturbations(background, np.array(np.float64(labels)))

f_fid = linear_perturbations_emu.growth_rate()

## Conversion of parameters
def conv_comet_to_pbj(pars_dict_comet,f_fid_arr):

    ck4_pbj_mono = - pars_dict_comet['cnlo']*(pars_dict_comet['b1']+5/7*f_fid_arr)/(pars_dict_comet['b1']**2+10/7*pars_dict_comet['b1']*f_fid_arr+5/9*f_fid_arr**2)
    ck4_pbj_quad = - pars_dict_comet['cnlo']*(pars_dict_comet['b1']+70/84*f_fid_arr)/(pars_dict_comet['b1']**2+140/84*pars_dict_comet['b1']*f_fid_arr+5/9*f_fid_arr**2)

    pars_dict_pbj = deepcopy(pars_dict_comet)

    #f_fid = np.array([0.87355692, 0.91199674, 0.94065152])

    pars_dict_pbj.update({
        'c0': pars_dict_comet['c0']-pars_dict_comet['c2']/2 + 3/8 * pars_dict_comet['c4'],
        'c2': 1 / f_fid_arr * (3/2 *pars_dict_comet['c2'] - 30/8 * pars_dict_comet['c4']),
        'c4': 1 / f_fid_arr**2 * 35/8 * pars_dict_comet['c4'],'mnu': 0.00001, 'N_mnu': 1,
        'bG3': pars_dict_comet['bGam3'],
        'ck4': (ck4_pbj_mono+ck4_pbj_quad)/2, # average of mono and quad contributions
    })
    pars_dict_pbj['ck4'] = -(ck4_pbj_mono+ck4_pbj_quad)

    return pars_dict_pbj


parameters_pbj = conv_comet_to_pbj(parameters,f_fid)

print("⏱️ Timing log-likelihood evaluation:")

start = time.time()
loglike_val = like_spec_pbj.loglike(parameters_pbj)
end = time.time()

print(f"Log-likelihood: {loglike_val}")
print(f"-2 * log-likelihood: {-2 * loglike_val}")
print(f"⏱️ Time elapsed: {end - start:.3f} seconds")

# AM priors for COMET (not used now)
AM_priors = {
    z: {'bGam3': [0.0, 50.0], 'c0': [0.0, 200.0], 'c2': [0.0, 200.0], 'c4': [0.0, 200.0], 'cnlo': [0.0, 4000.0], 'NP0':[1,2], 'NP20': [0.0, 30.0], 'NP22': [0.0, 30.0]} for z in labels}
# Approximate conversion to PBJ
AM_priors_pbj = {
    z: {'bG3': [0.0, 5.0], 'c0': [0.0, 100.0], 'c2': [0.0, 200.0], 'c4': [0.0, 750.0], 'ck4': [0.0, 2000.0], 'NP0':[1,2], 'NP20': [0.0, 30.0], 'NP22': [0.0, 30.0]} for z in labels}
    #z: {'bG3': [0.0, 50.0], 'c0': [0.0, 200.0], 'c2': [0.0, 300.0], 'c4': [0.0, 1500.0], 'ck4': [0.0, 2000.0], 'NP0':[1,2], 'NP20': [0.0, 30.0], 'NP22': [0.0, 30.0]} for z in labels}
    #z: {'bG3': [0.0, 10.0], 'c0': [0.0, 100.0], 'c2': [30.0, 300.0], 'c4': [0.0, 300.0], 'ck4': [0.0, 1500.0], 'NP0':[1,2], 'NP20': [0.0, 10.0], 'NP22': [0.0, 10.0]} for z in labels}
like_AM_pbj = EuclidLikelihood_GCspectro_Pls(data=data, settings=settings, Background=CAMBBackground, SpectroPower=PBJSpectroPower, Perturbations=HMemuLinearPerturbations, AM_priors=AM_priors_pbj)

print (-2*like_AM_pbj.loglike_AM(parameters_pbj, use_Jeffreys=True))

# ============================================================================
# 4. DEFINE PRIOR DISTRIBUTION FOR NAUTILUS
# ============================================================================

print("Setting up prior distribution...")
prior = Prior()

# Add parameters to the prior
# Format: prior.add_parameter(name, dist=(min, max)) for uniform
#         prior.add_parameter(name, dist=norm(loc=mean, scale=std)) for Gaussian

# Cosmological parameters (wide priors)
#prior.add_parameter('ombh2', dist=norm(loc=0.02237, scale=0.00055))
#prior.add_parameter('ombh2', dist=norm(loc=0.02268, scale=0.00038))
prior.add_parameter('ombh2', dist=norm(loc=0.0219961, scale=0.00055))
prior.add_parameter('omch2', dist=(0.05, 0.20))
#prior.add_parameter('logAs', dist=(2,4))
prior.add_parameter('As', dist=(0.5e-9, 3.0e-9))
prior.add_parameter('ns', dist=norm(loc=0.9649, scale=0.042))
prior.add_parameter('H0', dist=(50, 100))


prior.add_parameter('w0', dist=(-3.0, -0.3))
prior.add_parameter('wa', dist=(-3.0, 3.0))


# Galaxy bias parameters
prior.add_parameter('b1_z1', dist=(0.1, 4.0))
prior.add_parameter('b1_z2', dist=(0.1, 4.0))
prior.add_parameter('b1_z3', dist=(0.1, 4.0))


prior.add_parameter('b2_z1', dist=(-5.0, 5.0))
prior.add_parameter('b2_z2', dist=(-5.0, 5.0))
prior.add_parameter('b2_z3', dist=(-5.0, 5.0))
prior.add_parameter('bG2_z1', dist=(-4.0, 4.0))
prior.add_parameter('bG2_z2', dist=(-4.0, 4.0))
prior.add_parameter('bG2_z3', dist=(-4.0, 4.0))



#prior.add_parameter('b2_z1', dist=norm(loc=0., scale=10))
#prior.add_parameter('b2_z2', dist=norm(loc=0., scale=10))
#prior.add_parameter('b2_z3', dist=norm(loc=0., scale=10))
#prior.add_parameter('bG2_z1', dist=norm(loc=0, scale=10))
#prior.add_parameter('bG2_z2', dist=norm(loc=0, scale=10))
#prior.add_parameter('bG2_z3', dist=norm(loc=0, scale=10))

# prior.add_parameter('b2_z1', dist=norm(loc=0., scale=1))
# prior.add_parameter('b2_z2', dist=norm(loc=0., scale=1))
# prior.add_parameter('b2_z3', dist=norm(loc=0., scale=1))
# prior.add_parameter('bG2_z1', dist=norm(loc=0, scale=1))
# prior.add_parameter('bG2_z2', dist=norm(loc=0, scale=1))
# prior.add_parameter('bG2_z3', dist=norm(loc=0, scale=1))

print(f"Total number of sampled parameters: {prior.dimensionality()}")
print("All sampled parameters", prior.keys)

# ============================================================================
# 5. DEFINE LIKELIHOOD WRAPPER FOR NAUTILUS
# ============================================================================

def like_Nautilus(param_dict):
    """
    Wrapper function connecting Nautilus sampler to cloelike likelihood.
    
    This function:
    1. Takes parameters from Nautilus sampler
    2. Converts sampling parameters to cloelib base parameters
    3. Evaluates the likelihood
    4. Returns log-likelihood for Nautilus
    
    Args:
        param_dict: Dictionary of sampled parameters from Nautilus
        
    Returns:
        float: log-likelihood value (or -inf if evaluation fails)
    """
    
    # Start with default parameters
    pars = parameters_pbj.copy()
    
    # Update with sampled parameters
    pars.update(param_dict)
    if pars['w0'] + pars['wa'] >= 0:
        return -np.inf
    
    # IMPORTANT: Convert physical parameters to cloelib base parameters
    # Nautilus samples in convenient physical units, but cloelib expects
    # certain base parameters. This is the KEY CONVERSION STEP.
    pars['Omega_cdm0'] = param_dict['omch2'] / (pars['H0']/100)**2
    pars['Omega_b0'] = param_dict['ombh2'] / (pars['H0']/100)**2
    #pars['As'] = np.exp(param_dict['logAs']) * 1e-10
    pars['b1'] = np.array([param_dict['b1_z%d'%iz]for iz in range(1,4)])
    pars['b2'] = np.array([param_dict['b2_z%d'%iz]for iz in range(1,4)])
    pars['bG2'] = np.array([param_dict['bG2_z%d'%iz]for iz in range(1,4)])
    # Evaluate likelihood with cloelike
    try:
        log_likelihood = like_AM_pbj.loglike_AM(pars, use_Jeffreys=False, do_reparam=True)
    except (ValueError, RuntimeError):
        # If evaluation fails (e.g., unphysical parameters), return -inf
        log_likelihood = -np.inf
    
    return log_likelihood

# ====================================================================================
# 6. RUN NAUTILUS SAMPLER
#
# This will run the sampler and save results
#
# Note: The sampling process may take several minutes depending on the number
# of live points and the complexity of the likelihood. You can adjust the
# number of live points in the Sampler initialization for faster testing.
#
# For parallel runs, you can use the 'pool' argument in the Sampler to provide a multiprocessing pool.
# Check https://nautilus-sampler.readthedocs.io/en/latest/guides/parallelization.html
#
# ====================================================================================
def main():   
    print("=" * 60)
    print("Running Nautilus sampler...")
    print("=" * 60)

    # Initialize sampler with prior and likelihood
    sampler = Sampler(
        prior,
        like_Nautilus,
        n_live=2000,                           # Number of live points
        filepath='chains/hdf5/checkpoint_spec_pbj_DR1_like_tcm_andreapedro_priors_cpl_AapD2As.hdf5',
        pool=14       # Checkpoint file
    )

    # Run the sampler
    t_start = time.time()
    sampler.run(verbose=True)
    t_end = time.time()

    print()
    print(f"Sampling completed in {(t_end - t_start):.1f} seconds")
    print()

    # ============================================================================
    # 7. SAVE RESULTS
    # ============================================================================

    # Extract posterior samples
    points, log_w, log_l = sampler.posterior()

    # Save to compressed file
    np.savez_compressed(
        "chains/chain_spec_pbj_DR1_like_tcm_andreapedro_priors_cpl_AapD2As.npz",
        chain=points,           # Posterior samples
        weights=log_w,          # Log-weights
        logl=log_l              # Log-likelihood values

    )


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()