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



import HMcode2020Emu as hmcodeemu
HM2020_emu = hmcodeemu.Matter_powerspectrum()

# Data from table 1 of https://arxiv.org/pdf/2404.03002
DESI_DATA_SETS = {
    "BGS": {
        "z_eff": 0.295,
        "mean": 0.46474358, #0.3774, #change for synthetic tests
        "sigma": 0.09,
    },
    "LRG1": {
        "z_eff": 0.51,
        "mean": 0.46967991, #0.5145, #change for synthetic tests
        "sigma": 0.06,
    },
    "LRG2": {
        "z_eff": 0.706,
        "mean": 0.46206145, #0.4829,  #change for synthetic tests
        "sigma": 0.05,
    },
    "LRG3": {
        "z_eff": 0.919,
        "mean": 0.4450292, #0.4212,  #change for synthetic tests
        "sigma": 0.04,

    },
    "ELG2": {
        "z_eff": 1.317,
        "mean":  0.40256613, #0.3747, #change for synthetic tests
        "sigma": 0.035,

    },
    "QSO": {
        "z_eff": 1.491,
        "mean":  0.3832018, #0.4342, #change for synthetic tests
        "sigma": 0.04,
    },

    

}

class DESIGrowthLikelihood():
    """
    The DR2 2025 DESI likelihoods from https://arxiv.org/pdf/2503.14738

    We allow the user to specify which data sets to use, and combine
    them all into one. The data sets are:
    - BGS
    - LRG1
    - LRG2
    - LRG3+ELG1
    - ELG2
    - QSO
    - Lya

    The LRG3 and ELG1 data sets are also available separately, but
    cannot be used jointly with the LRG3+ELG1 data set. The default "all"
    selection leaves out the two individual data sets and just uses the combined one.
    """
    def __init__(self):
        data_sets = list(DESI_DATA_SETS.keys())
        self.data_sets = data_sets
        self.kind =  "cubic"


        self.data_x, self.data_y = self.build_data() 
        self.cov = self.build_covariance()
        self.inv_cov = np.linalg.inv(self.cov)

    

    def build_data(self):
        z = []
        mu = []
        kinds = []
        for name in self.data_sets:
            ds = DESI_DATA_SETS[name]

            # collect the effective redshfits for the measurements
            z.append(ds["z_eff"])
            mu.append(ds["mean"])


        z = np.array(z)
        mu = np.array(mu)
        print("data-vector is created")

        return z, mu

    def build_covariance(self):
        n = len(self.data_x)
        C = np.zeros((n, n))
        i = 0
        for name in self.data_sets:
            ds = DESI_DATA_SETS[name]
            C[i, i] = ds["sigma"]**2
            i += 1
        print("covariance is created")
        return C

    def extract_theory_points(self, param_dic):
        z_theory = self.data_x
        y = np.zeros(self.data_x.size)


        params_hm_emu = {
            "omega_cdm": param_dic['Omega_cdm0'],
            "omega_baryon": param_dic['Omega_b0'],
            "As": param_dic['As'],
            "ns": param_dic['ns'],
            "hubble": param_dic['H0'] / 100,
            "neutrino_mass": param_dic['mnu'],
            "w0": param_dic['w0'],
            "wa": param_dic['wa'],
        }

        hm_bounds = HM2020_emu.emulator["linear"]["bounds"]

        for key in params_hm_emu.keys():
            if np.prod(params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 lin emulator out of range.")
            else:
                params_hm_emu[key] = np.tile(params_hm_emu[key], len(z_theory))

        params_hm_emu["z"] = z_theory

        _, fsigma8 = HM2020_emu.get_sigma8(**params_hm_emu)
        y = fsigma8
        #print("y: ", y)    
        return y

    def do_likelihood(self, param_dic):
        #get data x by interpolation
        x = np.atleast_1d(self.extract_theory_points(param_dic))
        mu = np.atleast_1d(self.data_y)
        #gaussian likelihood
        d = x-mu
        chi2 = np.einsum('i,ij,j', d, self.inv_cov, d)
        chi2 = float(chi2)
        like = -0.5*chi2
        return like


hdf5_name = chain_name = 'checkpoint_desidr1_fsigma8_cmbpriors'



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

like_instance_desi_dr1 = DESIGrowthLikelihood()
loglike_desidr1 = like_instance_desi_dr1.do_likelihood(fiducial_cosmology)
print("✨ Log-likelihood at fiducial (desidr2): ", loglike_desidr1)

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
prior.add_parameter('H0', dist=(50, 90))
prior.add_parameter('logAs', norm(loc=np.log(1e10*fiducial_cosmology['As']), scale=0.016))
prior.add_parameter('ns', norm(loc=fiducial_cosmology['ns'], scale=0.0044))
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
    pars_in['As'] = np.exp(param_dict['logAs']) * 1e-10
    try:
        like = like_instance_desi_dr1.do_likelihood(pars_in)
    except ValueError:
        like = -np.inf

    return like



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

    
    # --- Save everything in a labelled way ---
    np.savez_compressed(
        'chains/chain_' + chain_name + '.npz',
        chain=chain_dict,
        weights=log_w,
        logl=log_l,
        param_names=param_names,
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
