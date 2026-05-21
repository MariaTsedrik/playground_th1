import numpy as np
from scipy.interpolate import interp1d
from cloelib.cosmology.wz_cosmology import DEBackground
#from cloelib.cosmology.camb_cosmology import CAMBBackground


# The three different types of measurement
# of BAO used in this data release
KIND_DV = 1
KIND_DM = 2
KIND_DH = 3


# Data from table 1 of https://arxiv.org/pdf/2404.03002
DESI_DATA_SETS = {
    "BGS": {
        "kind": "d_v",
        "z_eff": 0.295,
        "mean": 7.89638688, #7.944, #change for synthetic tests
        "sigma": 0.075,
    },
    "LRG1": {
        "kind": "d_m_d_h",
        "z_eff": 0.51,
        "mean": [13.23161186, 22.23768862], #[13.587, 21.863], #change for synthetic tests
        "sigma": [0.169, 0.427],
        "corr": -0.475,
    },
    "LRG2": {
        "kind": "d_m_d_h",
        "z_eff": 0.706,
        "mean": [17.35529559, 19.88509766], #[17.347, 19.458], #change for synthetic tests
        "sigma": [0.180, 0.332],
        "corr": -0.423,
    },
    "LRG3+ELG1": {
        "kind": "d_m_d_h",
        "z_eff": 0.934,
        "mean": [21.6072878, 17.46883317], #[21.574, 17.641], #change for synthetic tests
        "sigma": [0.153, 0.193],
        "corr": -0.425,
    },
    "ELG2": {
        "kind": "d_m_d_h",
        "z_eff": 1.321,
        "mean":  [27.6879227,  14.10356897], #[27.605, 14.178], #change for synthetic tests
        "sigma": [0.320, 0.217],
        "corr": -0.437,
    },
    "QSO": {
        "kind": "d_m_d_h",
        "z_eff": 1.484,
        "mean":  [29.88991661, 12.93756879], #[30.519,12.816], #change for synthetic tests
        "sigma": [0.758,0.513],
        "corr": -0.489
    },
    "Lya": {
        "kind": "d_m_d_h",
        "z_eff": 2.330,
        "mean": [38.84683454, 8.66249846],  #[38.988, 8.632], #change for synthetic tests
        "sigma": [0.531, 0.101],
        "corr": -0.431,
    },
    

}

class DESILikelihood():
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

            # The d_v type measurements are just a single number
            # but the d_m_d_h measurements are two values
            if ds["kind"] == "d_v":
                mu.append(ds["mean"])
                kinds.append(KIND_DV)
            else:
                mu.extend(ds["mean"])
                kinds.append(KIND_DM)
                kinds.append(KIND_DH)
                # This makes the z array the same length
                # as the mu array. But because the D_M and D_H
                # measurements are at the same redshift we only
                # need to store the redshift once, and this should
                # hopefully trigger an error if we mess up later.
                z.append(ds["z_eff"])

        kinds = np.array(kinds)
        z = np.array(z)
        mu = np.array(mu)

        # record the indices of the d_v and d_m_d_h measurements
        # for later
        self.dv_index = np.where(kinds==KIND_DV)[0]
        self.dm_index = np.where(kinds==KIND_DM)[0]
        self.dh_index = np.where(kinds==KIND_DH)[0]

        self.any_dv = len(self.dv_index) > 0
        self.any_dmdh = len(self.dm_index) > 0
        print("data-vector is created")

        return z, mu

    def build_covariance(self):
        n = len(self.data_x)
        C = np.zeros((n, n))
        i = 0
        for name in self.data_sets:
            ds = DESI_DATA_SETS[name]
            if ds["kind"] == "d_v":
                C[i, i] = ds["sigma"]**2
                i += 1
            else:
                C[i, i] = ds["sigma"][0]**2
                C[i+1, i+1] = ds["sigma"][1]**2
                C[i, i+1] = C[i+1, i] = ds["corr"]*ds["sigma"][0]*ds["sigma"][1]
                i += 2

        print("covariance is created")
        return C

    def extract_theory_points(self, param_dic):
        z_background = np.linspace(0., 4., 256)
        z_theory = z_background
        y = np.zeros(self.data_x.size)

        #print(param_dic)

        # Create background
        background = DEBackground(H0=param_dic["H0"],
                            Omega_cdm0=param_dic["Omega_cdm0"],
                            Omega_b0=param_dic["Omega_b0"],
                            Omega_k0=param_dic["Omega_k0"],
                            ns=param_dic["ns"],
                            As=param_dic["As"],
                            mnu=param_dic["mnu"],
                            zbin_edges = param_dic["zbin_edges"],
                            w_i = param_dic["w_i"],
                            binning = param_dic["binning"]

        )

        r_s = background.rdrag
        #print("r_s from camb: ", r_s)
        #w_nu = 0.0107
        #Obh2 = background.Omega_b0*background.h**2
        #Och2 = background.Omega_cdm0*background.h**2
        #rdrag= 55.154*np.exp(-72.3*(w_nu*background.mnu+0.0006)**2)/(Obh2**0.12807 * (Obh2+Och2)**0.25351)
        #print("r_s approximation: ", rdrag)
        #print("ratio: ", rdrag/r_s)



        D_C = background.comoving_distance(z_background)
        D_C = np.hstack(([0.], D_C))
        H = background.hubble_parameter(z_background, "1/Mpc")
        D_H = 1 / H[0]


        D_M = D_C

        D_L = D_M * (1 + z_background)
        D_A = D_M / (1 + z_background)
        D_V = ((1 + z_background)**2 * z_background * D_A**2 / H)**(1./3.)

        # Deal with mu(0), which is -np.inf
        #mu = np.zeros_like(D_L)
        #pos = D_L > 0
        #mu[pos] = 5*np.log10(D_L[pos])+25
        #mu[~pos] = -np.inf


        if self.any_dv:
            d_v = D_V
            z_data = self.data_x[self.dv_index]
            f = interp1d(z_theory, d_v/r_s, kind=self.kind)
            y[self.dv_index] = f(z_data)

        if self.any_dmdh:
            z_data = self.data_x[self.dm_index]

            d_m = D_M
            f = interp1d(z_theory, d_m/r_s, kind=self.kind)
            y[self.dm_index] = f(z_data)

            d_h = 1.0 / H
            f = interp1d(z_theory, d_h/r_s, kind=self.kind)
            y[self.dh_index] = f(z_data)

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

