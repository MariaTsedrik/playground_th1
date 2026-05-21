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


from desi_dr2_likelihood import *
from des_y5_sn_likelihood import *
# We read with euclidlib v2025.2 (v2025.1 is also compatible)


hdf5_name = chain_name = 'checkpoint_cmbpriors_desy5sn_real_v2' #'checkpoint_cmbpriors_desy5sn_desidr2bao'



fiducial_cosmology = {
    'H0': 67,                # Hubble constant
    'Omega_cdm0': 0.27,        # Cold dark matter density
    'Omega_b0': 0.049,         # Baryon density
    'Omega_k0': 0.0,           # Curvature
    'mnu': 0.077,                # Neutrino mass
    'w0': -0.75,                # Dark energy equation of state (present)
    'wa': -0.86,                 # Dark energy equation of state (evolution)
    'ns': 0.96,                # Scalar spectral index
    'As': 2.1e-9, # Scalar amplitude
    'gamma_MG': 0.545,         # Modified gravity parameter
    'log10TAGN': 7.75,          # AGN feedback parameter
    'N_mnu': 1,                  # Number of massive neutrino species
    'N_eff': 3.046             # Effective number of relativistic species
}



like_instance_desi_dr2 = DESILikelihood()
loglike_desidr2 = like_instance_desi_dr2.do_likelihood(fiducial_cosmology)
print("✨ Log-likelihood at fiducial (desidr2): ", loglike_desidr2)

like_instance_des_y5_sn = DESY5SNLikelihood()
loglike_desy5sn = like_instance_des_y5_sn.do_likelihood(fiducial_cosmology)
print("✨ Log-likelihood at fiducial (desy5sn): ", loglike_desy5sn)


# Set priors
prior = Prior()
#prior.add_parameter('Omega_cdm0', dist=(0.1, 0.8))
#prior.add_parameter('Omega_b0', dist=(0.01, 0.1))
#prior.add_parameter('H0', dist=(40., 100.))
#prior.add_parameter('logAs', dist=(np.log(0.5e-9*1e10), np.log(5e-9*1e10)))
#prior.add_parameter('ns', dist=(0.6, 1.2))


h_fid = fiducial_cosmology['H0']/100.
prior.add_parameter('omch2', norm(loc=fiducial_cosmology['Omega_cdm0']*h_fid*h_fid , scale=0.0014))
prior.add_parameter('ombh2', norm(loc=fiducial_cosmology['Omega_b0']*h_fid*h_fid , scale=0.00015))
#prior.add_parameter('H0', dist=(50, 90))
#prior.add_parameter('logAs', norm(loc=np.log(1e10*fiducial_cosmology['As']), scale=0.016))
#prior.add_parameter('ns', norm(loc=fiducial_cosmology['ns'], scale=0.0044))
prior.add_parameter('w0', dist=(-3.0, -0.3))
prior.add_parameter('wa', dist=(-3.0, 3.0))

print("✨ Prior set up completed.")
print("Parameters sampled: ", prior.keys)
print("Number of sampled parameters:", len(prior.keys))

default_pars = fiducial_cosmology 

def like_Naut_test(param_dict):
    # Enforce physical prior on w0 + wa
    if param_dict['w0'] + param_dict['wa'] >= 0:
        return -np.inf

    pars_in = default_pars.copy()
    pars_in.update(param_dict)

    h = pars_in['H0'] / 100.0
    # need these parameters for cloe
    # uncomment for cmb-priors!!!
    pars_in['Omega_cdm0'] = param_dict['omch2'] / h**2
    pars_in['Omega_b0'] = param_dict['ombh2'] / h**2
    #pars_in['As'] = np.exp(param_dict['logAs']) * 1e-10
    try:
        #like = like_instance_desi_dr2.do_likelihood(pars_in)
        like = like_instance_des_y5_sn.do_likelihood(pars_in)
        #like = like_instance_desi_dr2.do_likelihood(pars_in) + like_instance_des_y5_sn.do_likelihood(pars_in)
    except ValueError:
        like = -np.inf

    return like

#blob_vec     = [('sigma8_0', float)]

parallelisation_option = "MPI"
sampling_workers = 32

if __name__ == '__main__':
    # Initialize Nautilus sampler 
    sampler = Sampler(
        prior, 
        like_Naut_test, 
        n_live=5000,
        filepath='chains/hdf5/'+hdf5_name+'.hdf5', 
        resume=True, 
        pool=MPIPoolExecutor()  # MPI-based parallelization
    )
    start = time.time()
    sampler.run(verbose=True, discard_exploration=True)
    finish = time.time()
    chain_time = finish - start

    # Extract results
    log_z = sampler.log_z
    points, log_w, log_l = sampler.posterior()
    #points, log_w, log_l, derived = sampler.posterior(return_blobs=True)


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

    #print('derived: ', derived)

    # --- Derived parameters ---
    #derived_names = [name for name, _ in blob_vec]
    #print('derived names: ', derived_names)

    #derived_dict = {
    #    derived_names[0]: np.array(derived)
    #}
    
    # --- Save everything in a labelled way ---
    np.savez_compressed(
        'chains/chain_' + chain_name + '.npz',
        chain=chain_dict,
        #derived=derived_dict,
        weights=log_w,
        logl=log_l,
        param_names=param_names,
        #derived_names=derived_names,
        logz=log_z,
        chain_time=chain_time,
        chain_time_hms=chain_time_hms,
        fiducial=default_pars
    )

    #np.savetxt(
    #    "chains/chain_"+chain_name+".txt", 
    #    np.c_[points, log_w, log_l], 
    #    header=output_header, 
    #    footer=footer
    #)

    print(f"{parallelisation_option} with {sampling_workers} processes: sampling complete!")
    print(f"  Evidence: log_Z = {log_z:.6f}")
    print(f"  Total time: {chain_time:.2f} s ({chain_time_hms})")
    print(f"  Output files:")
    print(f"    HDF5: chains/hdf5/{hdf5_name}.hdf5")
    print(f"    Chain: chains/chain_{chain_name}.npz")


