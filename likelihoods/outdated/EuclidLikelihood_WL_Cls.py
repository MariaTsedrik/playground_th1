import numpy as np

from copy import deepcopy
from dataclasses import replace

from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.cosmology.ReACTEmu_cosmology import MGemuNonlinearBoost, BoostedPerturbations
#from cloelib.cosmology.wz_cosmology import DEBackground, DELinearPerturbations, DENonlinearPerturbations
from cloelib.cosmology.MGrowth_cosmology import MGrowthLinearPerturbations

zfinal = 1000.
zmax = 3.
zmin = 0.
zbin_edges = np.array([0., 0.4, 0.8, 1.2, 1.6, 2., zmax, zfinal])
zbin_centers = 0.5*(zbin_edges[1:] + zbin_edges[:-1])
zbin_widths = np.diff(zbin_edges)

class EuclidLikelihood_WL_Cls:
    def __init__(
        self,
        data: dict,
        settings: dict,
        Background: type,
        LinPerturbations: type,
        NonLinPerturbations: type,
        BackgroundLCDM=None,
        LinPerturbationsBase=None, #type
        gravity_model=None #str
    ):
        r"""Class constructor
        Parameters
        ----------
        data: dict
            Data dictionary
        settings: dict
            Settings dictionary
        Background: type
            Protocol-consistent Background class type
        LinPerturbations: type
            Protocol-consistent Perturbations class type
        NonLinPerturbations: type
            Protocol-consistent Perturbation class type
        """
        self.data = data
        self.settings = settings
        self.model = None

        self.Background = Background
        self.LinPerturbations = LinPerturbations
        if NonLinPerturbations == HMemuNonLinearPerturbations:
            self.NonLinPerturbations = NonLinPerturbations
            self.model = 'hmcode'
        elif NonLinPerturbations == None:
            self.model = 'mgrowth'
            self.LinPerturbationsBase = LinPerturbationsBase
            self.gravity_model = gravity_model
        elif NonLinPerturbations == BoostedPerturbations:
            self.NonLinPerturbations = NonLinPerturbations
            self.model = 'mgemu'
            self.LinPerturbationsBase = LinPerturbationsBase
            self.gravity_model = gravity_model
            self.NonlinPerturbationsBase=HMemuNonLinearPerturbations
        # elif NonLinPerturbations == DENonlinearPerturbations:
        #     self.NonLinPerturbations = NonLinPerturbations
        #     self.model = 'de_bin'
        #     self.LinPerturbationsBase = LinPerturbationsBase
        #     self.BackgroundLCDM = BackgroundLCDM

        else:
            raise TypeError(
                "Currenty, this only works for the HMcode "
                "emulator and MGrowth and MGEmus, so NonLinPerturbations must be of "
                "type HMemuNonLinearPerturbations or None or BoostedPerturbations"
            )

        self.scale_cuts = settings["scale_cuts"]

        self.rebin = False
        self.zs = data["z_arr"]
        self.mixmat = deepcopy(data["mixmat"])

        self.n_she_bins = self.data["dndz_she"].shape[0]
        self.n_she_bins = self.data["dndz_she"].shape[0]
        IA_keys = ["AIA", "EtaIA"]
        mul_bias_keys = [
            "multiplicative_bias_%d" % i for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        width_she_keys = [f"width_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys + width_she_keys

        self.WL_keys = []
        for i in range(1, self.n_she_bins + 1):
            for j in range(i, self.n_she_bins + 1):
                self.WL_keys.append(("SHE", "SHE", i, j))

        self._prepare()

    def _prepare(self):
        r"""Arrange data vectors and covariance matrices in format required
        for :math:`\chi^2` calculation
        """
        if self.settings["n_ell_bins"] < len(self.data["ells"]):
            self.rebin = True
            self.data["cells_unbin"] = self.data["cells"]
            self.data["ells_unbin"] = self.data["ells"]
            self._bin_data(
                self.data["cells"], self.data["ells"], self.settings["n_ell_bins"]
            )
            self._bin_mixmat()

        self._flatten_data_vector_and_mask()
        self._mask_covariance_and_invert()

    def _bin_data(self, cells_data, ells, n_bins):
        r"""Rebin WL photometric data
        Parameters
        ----------
        Returns
        -------
        """
        bin_edges = np.geomspace(10, ells[-1], n_bins + 1)
        mask_bins = [
            (ells > bin_edges[i]) & (ells < bin_edges[i + 1]) for i in range(n_bins)
        ]
        self.weight_mat = np.asarray(mask_bins, dtype=float)
        self.weight_mat /= np.sum(self.weight_mat, axis=1)[:, None]

        cells_ave = {k: (cells_data[k] @ self.weight_mat.T) for k in cells_data.keys()}
        self.data["ells"] = np.array([np.mean(ells[mb]) for mb in mask_bins])
        self.data["cells"] = cells_ave

    def _bin_mixmat(self):
        r"""Rebin WL mixing matrix"""
        # self.mixmat = {k: np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
        #                for k in self.mixmat}
        for k in self.mixmat.keys():
            new_array = np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
            if k[:2] == ("SHE", "SHE"):
                new_array = np.transpose(new_array, axes=(1, 0, 2))

            self.mixmat[k] = replace(self.mixmat[k], array=new_array)

    def _masking(self, arr: np.ndarray, interval: list) -> np.ndarray:
        r"""Get a 1/0 mask for the elements of arr contained in interval
        Parameters
        ----------
        arr: numpy.ndarray
            Input array
        interval: list
            Edges defining the masking region
        Returns
        -------
        masked_arr: numpy.ndarray
            Masked array
        """
        return (arr >= interval[0]) & (arr <= interval[1])

    def _flatten_data_vector_and_mask(self):
        r"""Arranges data vectors into flattened vectors and mask them"""
        self.data_vector = np.transpose(
            [self.data["cells"][key][:2] for key in self.WL_keys], axes=(1, 0, 2)
        ).flatten()

        self.masking_vector = np.transpose(
            [
                [
                    self._masking(self.data["ells"], self.scale_cuts[key][i])
                    for i in [0, 1]
                ]
                for key in self.WL_keys
            ],
            axes=(1, 0, 2),
        ).flatten()

        self.masked_data_vector = self.data_vector[self.masking_vector]

    def _mask_covariance_and_invert(self):
        r"""Arrange, mask, and invert covariance matrices"""
        self.covariance_matrix = self.data["cov"]

        self.inverse_masked_covariance_matrix = np.linalg.inv(
            self.covariance_matrix[self.masking_vector][:, self.masking_vector]
        )

    def get_theory_vector(self, parameters: dict):
        r"""Generate theory vectors based on specified parameters
        Parameters
        ----------
        parameters: dict
            Input parameters
        Return
        ------
        theory_vector: dict
            Stacked theory vectors
        """
        if self.model == 'hmcode':
            background = self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=parameters["w0"],
            wa=parameters["wa"],
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
            N_mnu=parameters["N_mnu"],
            )
            lp = self.LinPerturbations(background, self.zs)
            nlp = self.NonLinPerturbations(
                background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
            )
        elif self.model == 'mgrowth': 
            background = self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=parameters["w0"],
            wa=parameters["wa"],
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
            N_mnu=parameters["N_mnu"]
            )
            base = self.LinPerturbationsBase(background, self.zs)
            nlp = self.LinPerturbations(background, base,
                                        gravity_model=self.gravity_model,
                                        mgpars=parameters) 
            if nlp.check_ranges==False:
                return None, False
        elif self.model == 'mgemu':
            background = self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=parameters["w0"],
            wa=parameters["wa"],
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
            N_mnu=parameters["N_mnu"],
            )
            background_lcdm = self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=-1.0,
            wa=0.0,
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
            N_mnu=parameters["N_mnu"],
            )
            # CAMB w0waCDM linear
            lp_base = self.LinPerturbationsBase(background, self.zs)
            # MGrowth cosmology (can have w0waCDM background)                                      
            lp = self.LinPerturbations(background, lp_base,
                                        gravity_model=self.gravity_model,
                                        mgpars=parameters) 
            # Nonlinear must be LCDM     
            lp_base_lcdm = self.LinPerturbationsBase(background_lcdm, self.zs)                    
            nlp_base = HMemuNonLinearPerturbations(background_lcdm, lp_base_lcdm, self.zs, 
                                                   log10TAGN=parameters["log10TAGN"])          
            # lp_base is needed only to check k and z ranges                                                        
            boost = MGemuNonlinearBoost(background, lp_base, self.zs, gravity_model=self.gravity_model, mgpars=parameters)
            if boost.check_ranges==False:
                return None, False
            # lp must be MGrowth for growth_factor and nlp_base must be LCDM    
            nlp = self.NonLinPerturbations(lp, nlp_base, boost.MGboost_interp)

        elif self.model == 'de_bin':
            w_i = np.array([parameters[f"wbin_{i}"] for i in range(1, 8)])
            background_lcdm = self.BackgroundLCDM(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=-1.0,
            wa=0.0,
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
            N_mnu=parameters["N_mnu"],
            )
            background = self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            ns=parameters["ns"],
            As=parameters["As"],
            mnu=parameters["mnu"],
            zbin_edges = zbin_edges,
            zbin_widths = zbin_widths,
            zbin_centers = zbin_centers,
            w_i = w_i
            )
            # CAMB w0waCDM linear
            linear_perturbations_lcdm = self.LinPerturbationsBase(background_lcdm, self.zs)
            # MGrowth cosmology (can have w0waCDM background)
            linear_perturbations_de = DELinearPerturbations(background, linear_perturbations_lcdm,
                                                        zbin_edges = zbin_edges,
                                                        zbin_widths = zbin_widths,
                                                        zbin_centers = zbin_centers,
                                                        w_i = w_i
                                                        )
            nonlinear_perturbations_de = DENonlinearPerturbations(background, linear_perturbations_de, self.zs, log10TAGN=parameters["log10TAGN"])
            nlp = nonlinear_perturbations_de

        she = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_she_keys}
            | {"CIA": 0.0134},
        )

        

        cell_all_th = {**AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat)}
        theory_vector = np.transpose(
            # [cell_all_th[key][:2] for key in self.WL_keys], axes=(1, 0, 2)
            [cell_all_th[key][0] for key in self.WL_keys], axes=(1, 0, 2)
        ).flatten()

        #cell_all_th = AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
        #theory_vector = np.array([cell_all_th[key][0, 0] for key in self.WL_keys]).flatten()
        #print('cell_all_th: ', cell_all_th)
        #print('theory_vector: ', theory_vector, theory_vector.shape)
        return theory_vector, True

    def _mask_theory_vector(self):
        r"""Mask theory vector"""
        #print('masking_vector: ', self.masking_vector, self.masking_vector.shape)
        self.masked_theory_vector = self.theory_vector[self.masking_vector]

    def loglike(self, parameters: dict):
        r"""Log-likelihood of WL probe
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        loglike: float
            Log-likelihood
        """
        self.theory_vector, status = self.get_theory_vector(parameters)
        if status == False or np.isnan(self.theory_vector).any():
            return -np.inf
        self._mask_theory_vector()

        diff = self.masked_theory_vector - self.masked_data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_masked_covariance_matrix), diff)

        return -0.5 * chi2
