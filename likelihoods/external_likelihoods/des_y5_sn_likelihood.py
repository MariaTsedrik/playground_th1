import os
import numpy as np
#from astropy.table import Table
import pandas as pd
from cloelib.cosmology.camb_cosmology import CAMBBackground

w0 = -0.75
wa = -0.86
fiducial_cosmology_sn = {
    'H0': 67,                # Hubble constant
    'Omega_cdm0': 0.27,        # Cold dark matter density
    'Omega_b0': 0.049,         # Baryon density
    'Omega_k0': 0.0,           # Curvature
    'mnu': 0.077,                # Neutrino mass
    'w0': w0,                # Dark energy equation of state (present)
    'wa': wa,                 # Dark energy equation of state (evolution)
    'ns': 0.96,                # Scalar spectral index
    'As': 2.1e-9, # Scalar amplitude
    'gamma_MG': 0.545,         # Modified gravity parameter
    'log10TAGN': 7.75,          # AGN feedback parameter
    'N_mnu': 1,                  # Number of massive neutrino species
   # 'N_eff': 3.046             # Effective number of relativistic species
}

default_data_file = "/leonardo/home/userexternal/mtsedrik/cosmosis-standard-library/likelihood/des-sn/DES-SN5YR_HD.csv"
default_covmat_file = "/leonardo/home/userexternal/mtsedrik/cosmosis-standard-library/likelihood/des-sn/STAT+SYS.txt.gz"

def cov_log_likelihood(mu_model, mu, inv_cov):
    """ 
    Computes modified likelihood computation to marginalize offset M 
    from https://arxiv.org/abs/astro-ph/0104009v1, see Equation A9-12 
    """
    delta = np.array([mu_model - mu])
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))
    return -0.5*chi2

class DESY5SNLikelihood():
    def __init__(self):
        self.kind =  "cubic"


        self.data_x, self.data_y = self.build_data() 
        self.cov = self.build_covariance()
        self.inv_cov = np.linalg.inv(self.cov)

    def build_data(self):
        """
        Run once at the start to load in the data vectors.

        Returns x, y where x is the independent variable (redshift in this case)
        and y is the Gaussian-distribured measured variable (magnitude in this case).

        """
        filename = default_data_file
        print("Loading DES Y5 SN data from {}".format(filename))
        data = pd.read_csv(filename) #Table.read(filename, format='ascii.csv')
        self.origlen = len(data)

        # The only columns that we actually need here are the redshift,
        # distance modulus and distance modulus error
        self.ww = (data['zHD']>0.00) 

        #use the vpec corrected redshift for zCMB 
        self.zCMB = data['zHD'][self.ww] 
        self.zHEL = data['zHEL'][self.ww]

        # distance modulus and relative stat uncertainties
        self.mu_obs = data['MU'][self.ww]
        self.mu_obs_err = data['MUERR_FINAL'][self.ww]

        #self.mu_obs = self.extract_theory_points(fiducial_cosmology_sn)
        # Return this to the parent class, which will use it
        # when working out the likelihood
        print(f"Found {len(self.zCMB)} DES SN 5 supernovae (or bins if you used the binned data file)")
        return self.zCMB, self.mu_obs

    def build_covariance(self):
        """Run once at the start to build the covariance matrix for the data"""
        filename = default_covmat_file
        print("Loading DESY5 SN covariance from {}".format(filename))

        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This data file is just the systematic component of the covariance - 
        # we also need to add in the statistical error on the magnitudes
        # that we loaded earlier
        d = np.loadtxt(filename)
        n = int(d[0])
        C = np.reshape(d[1:], (n,n))

        # Now add in the statistical error to the diagonal
        for i in range(n):
            C[i,i] += self.mu_obs_err[i]**2

        # Return the covariance; the parent class knows to invert this
        # later to get the precision matrix that we need for the likelihood.
        C = C[self.ww][:, self.ww]

        return C

    def extract_theory_points(self, param_dic):
        """
        Run once per parameter set to extract the mean vector that our
        data points are compared to. 
        """
        import scipy.interpolate
        z_background = np.linspace(0., 4., 256)

        # Create background
        background = CAMBBackground(
            H0=param_dic['H0'],
            Omega_cdm0=param_dic['Omega_cdm0'],
            Omega_b0=param_dic['Omega_b0'],
            w0=param_dic['w0'],
            wa=param_dic['wa'],
            Omega_k0=param_dic['Omega_k0'],
            ns=param_dic['ns'],
            As=param_dic['As'],
            mnu=param_dic['mnu'],
            gamma_MG=param_dic['gamma_MG'],
            N_mnu=param_dic['N_mnu']
        )

        # Pull out theory DA and z from the block.
        theory_x = z_background
        D_C = background.comoving_distance(z_background)
        D_A = D_C / (1 + z_background)
        theory_y = D_A

        # Interpolation function of theory so we can evaluate at redshifts of the data
        f = scipy.interpolate.interp1d(theory_x, theory_y, kind=self.kind)
        
        zcmb = self.zCMB
        zhel = self.zHEL

        # distance modulus
        theory_ynew = 5.0 * np.log10((1.0 + zcmb) * (1.0 + zhel) * np.atleast_1d(f(zcmb))) + 25.
        #print('theory_new: ', theory_ynew)
        # The offset M will be marginalized in the modified log likelihood computation
        return theory_ynew

    def do_likelihood(self, param_dic):
        # get data x by interpolation
        x = np.atleast_1d(self.extract_theory_points(param_dic))
        mu = np.atleast_1d(self.data_y)

        # modified log-likelihood computation to marginalize offset M
        like = cov_log_likelihood(x, mu, self.inv_cov)
        chi2 = -2.0*like
        like = float(like)
        return like



#like_instance_desy5sn = DESY5SNLikelihood()
# loglike = like_instance_desy5sn.do_likelihood(fiducial_cosmology)
# print("✨ Log-likelihood at fiducial: ", loglike)
# fiducial_cosmology['w0'] = -1.
# fiducial_cosmology['wa'] = 0.
# loglike = like_instance_desy5sn.do_likelihood(fiducial_cosmology)
# print("✨ Log-likelihood at fiducial: ", loglike)
# fiducial_cosmology['w0'] = -1.3
# fiducial_cosmology['wa'] = 0.
# loglike = like_instance_desy5sn.do_likelihood(fiducial_cosmology)
# print("✨ Log-likelihood at fiducial: ", loglike)