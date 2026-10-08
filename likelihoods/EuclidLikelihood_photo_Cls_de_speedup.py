import numpy as np
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations


def _freeze_value(value):
    """Return a form of ``value`` that compares equal with ``==``.

    Numpy arrays do not, so they are stored as raw bytes. Lists and dicts
    are walked so redshift edges and ``w_i`` participate in the cache key.
    """
    if isinstance(value, np.ndarray):
        return (value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _freeze_value(item)) for key, item in value.items())
        )
    return value


# Probe nuisance parameters. Everything else in ``parameters`` can change the
# background or the nonlinear spectrum, so it belongs in the cache key.
_NUISANCE_KEYS = frozenset({"AIA", "EtaIA", "CIA"})
_NUISANCE_PREFIXES = (
    "multiplicative_bias_",
    "dz_shear_",
    "width_shear_",
    "magnification_bias_",
    "dz_pos_",
    "width_pos_",
    #"b1_photo_poly",
    "b1_photo_bin",
)


def _is_nuisance_key(key):
    return key in _NUISANCE_KEYS or key.startswith(_NUISANCE_PREFIXES)


class _SharedDEPerturbations:
    """Build the binned-w background and perturbations once per cosmology.

    WL, GCph and GGL each implement ``get_theory_vector_full``, and a combined
    likelihood runs every mixin in the method-resolution order. Without a
    shared cache, 2x2pt builds the cosmology twice and 3x2pt builds it three
    times. WL-only still builds it once. The same cache also keeps the
    position and shear tracers, which GGL otherwise rebuilds on top of GCph
    and WL.
    """

    def _cosmo_key(self, parameters):
        return tuple(
            (key, _freeze_value(parameters[key]))
            for key in sorted(parameters)
            if not _is_nuisance_key(key)
        )

    def _get_perturbations(self, parameters):
        key = self._cosmo_key(parameters)
        cached = getattr(self, "_perturbations_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        background = self.Background(
            parameters,
            w_i=parameters["w_i"],
            zbin_edges=parameters["zbin_edges"],
        )
        background_lcdm = CAMBBackground(
            H0=background.H0,
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
        linear_perturbations_lcdm = HMemuLinearPerturbations(background_lcdm, self.zs)
        lp = self.LinPerturbations(background, linear_perturbations_lcdm)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )
        # Computed once per cosmology. Calling it again from each mixin would
        # repeat the same integral on the shared perturbations object.
        self.derived["sigma8_0"] = nlp.sigma8_0()
        result = (background, lp, nlp)
        self._perturbations_cache = (key, result)
        return result

    def _get_pos_tracer(self, parameters, nlp):
        key = self._cosmo_key(parameters) + tuple(
            _freeze_value(parameters[name]) for name in self.full_pos_keys
        )
        cached = getattr(self, "_pos_tracer_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tracer = PositionsTracer(
            nlp,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params={name: parameters[name] for name in self.full_pos_keys},
            #galaxy_bias_model="poly",
            galaxy_bias_model="per_bin",
        )
        self._pos_tracer_cache = (key, tracer)
        return tracer

    def _get_she_tracer(self, parameters, nlp):
        key = self._cosmo_key(parameters) + tuple(
            _freeze_value(parameters[name]) for name in self.full_she_keys
        )
        cached = getattr(self, "_she_tracer_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tracer = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            ia_model="NLA",
            nuisance_params={name: parameters[name] for name in self.full_she_keys},
        )
        self._she_tracer_cache = (key, tracer)
        return tracer


class WLMixin:
    """
    Mixin class providing weak lensing (WL) specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_wl()

    def _init_wl(self):
        self.n_she_bins = self.data["dndz_she"].shape[0]
        IA_keys = ["AIA", "EtaIA", "CIA"]
        mul_bias_keys = [
            f"multiplicative_bias_{i}" for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        width_she_keys = [f"width_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys + width_she_keys
        self.WL_keys = [
            ("SHE", "SHE", i, j)
            for i in range(1, self.n_she_bins + 1)
            for j in range(i, self.n_she_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.WL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array(
            [self.data["cells"][key][0, 0] for key in self.WL_keys]
        ).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        she = self._get_she_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0, 0] for key in self.WL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(she, she).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0, 0] for key in self.WL_keys]).flatten()
        self.theory_prediction = cell_all_th
        return np.concatenate([v, vec])


class GCphMixin:
    """
    Mixin class providing photometric angular galaxy clustering specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gcph()

    def _init_gcph(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        #bias_keys = [f"b1_photo_poly{i}" for i in range(4)]
        bias_keys = [f"b1_photo_bin{i}" for i in range(0, self.n_pos_bins)]
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        width_pos_keys = [f"width_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys + width_pos_keys
        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(i, self.n_pos_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GG_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key] for key in self.GG_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        pos = self._get_pos_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, pos).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key] for key in self.GG_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, pos).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key] for key in self.GG_keys]).flatten()
        self.theory_prediction = cell_all_th
        return np.concatenate([v, vec])


class GGLMixin:
    """
    Mixin class providing galaxy-galaxy lensing specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_ggl()

    def _init_ggl(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        self.n_she_bins = self.data["dndz_she"].shape[0]
        #bias_keys = [f"b1_photo_poly{i}" for i in range(4)]
        bias_keys = [f"b1_photo_bin{i}" for i in range(0, self.n_pos_bins)]
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        width_pos_keys = [f"width_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        IA_keys = ["AIA", "EtaIA", "CIA"]
        mul_bias_keys = [
            f"multiplicative_bias_{i}" for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        width_she_keys = [f"width_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys + width_pos_keys
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys + width_she_keys
        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(1, self.n_she_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GGL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key][0] for key in self.GGL_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        pos = self._get_pos_tracer(parameters, nlp)
        she = self._get_she_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0] for key in self.GGL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, she).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0] for key in self.GGL_keys]).flatten()
        self.theory_prediction = cell_all_th
        return np.concatenate([v, vec])


class EuclidLikelihood_WL(WLMixin, _SharedDEPerturbations, PhotoLikelihoodBase):
    """
    EuclidLikelihood_WL computes the weak lensing (WL) likelihood for photometric surveys using Euclid data.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        WLMixin: Mixin providing weak lensing specific functionality.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation, including observed ells and other relevant quantities.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    mode : str, optional
        Mode of operation, default is "coupled".
    """

    pass


class EuclidLikelihood_GCph(GCphMixin, _SharedDEPerturbations, PhotoLikelihoodBase):
    """
    EuclidLikelihood_GCph computes the likelihood for galaxy clustering photometric (GCph) data
    using the Euclid survey specifications.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing GCph-specific functionality.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation, including observed ells and other relevant quantities.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    mode : str, optional
        Mode of operation, default is "coupled".
    """

    pass


class EuclidLikelihood_GGL(GGLMixin, _SharedDEPerturbations, PhotoLikelihoodBase):
    """
    EuclidLikelihood_GGL class for galaxy-galaxy lensing likelihood computation.

    This class combines the functionalities of PhotoLikelihoodBase and GGLMixin to compute
    the likelihood for galaxy-galaxy lensing (GGL) using photometric data. It initializes
    the necessary components and constructs a masking vector based on scale cuts for each
    GGL key.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GGLMixin: Mixin providing GGL-specific functionality.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation, including observed ells and other relevant quantities.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    mode : str, optional
        Mode of operation, default is "coupled".
    """

    pass


class EuclidLikelihood_3x2pt(
    GCphMixin, GGLMixin, WLMixin, _SharedDEPerturbations, PhotoLikelihoodBase
):
    """
    EuclidLikelihood_3x2pt combines weak lensing (WL), galaxy clustering (GCph), and galaxy-galaxy lensing (GGL)
    likelihoods for photometric cosmological analyses, supporting scale cuts and masking.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing galaxy clustering specific functionality.
        GGLMixin: Mixin providing galaxy-galaxy lensing specific functionality.
        WLMixin: Mixin providing weak lensing specific functionality.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Also, the order of the mixins assumes
    the ordering of the covariance matrix blocks is GCph, GGL and WL.
    ``_SharedDEPerturbations`` sits immediately before ``PhotoLikelihoodBase`` so the
    background and nonlinear spectrum are built once and reused by every probe.

    Parameters
    ----------
    data : dict
        Dictionary containing observational data vectors and related metadata.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance providing background cosmology calculations.
    LinPerturbations : object
        Instance for linear perturbation theory calculations.
    NonLinPerturbations : object
        Instance for non-linear perturbation theory calculations.
    mode : str, optional
        Mode for likelihood calculation, default is "coupled".
    """

    pass


class EuclidLikelihood_2x2pt(
    GCphMixin, GGLMixin, _SharedDEPerturbations, PhotoLikelihoodBase
):
    """
    Likelihood class for Euclid 2x2pt photometric clustering and galaxy-galaxy lensing analysis.

    This class combines galaxy clustering (GCph) and galaxy-galaxy lensing (GGL) likelihoods,
    providing methods to compute the full data and theory vectors for both probes.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Also, the order of the mixins assumes
    the ordering of the covariance matrix blocks is GCph, GGL.
    ``_SharedDEPerturbations`` sits immediately before ``PhotoLikelihoodBase`` so the
    background and nonlinear spectrum are built once and reused by both probes.

    Parameters
    ----------
    data : dict
        Input data dictionary containing observed power spectra and related quantities.
    settings : dict
        Configuration settings for the likelihood analysis.
    Background : object
        Cosmological background model instance.
    LinPerturbations : object
        Linear perturbations model instance.
    NonLinPerturbations : object
        Non-linear perturbations model instance.
    mode : str, optional
        Mode for likelihood computation, default is "coupled".
    """

    pass
