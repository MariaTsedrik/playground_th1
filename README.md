# 🛝 Playground Repository

Welcome to the **Playground** repository of the [`cloe` organisation](https://github.com/cloe-org)! 🚀

This repository serves as a sandbox for tutorials, exercises, and validation for various features, scripts, and models related to `cloelib` and `cloelike`. It provides an open space to learn quickly how to get around the `cloe` organisation.

To explore the contents of this repository, you may need to download synthetic example data available at [Zenodo – cloe-org Community](https://zenodo.org/communities/cloe-org/records).

The data can be read using the [`euclidlib`](https://euclidlib.readthedocs.io/en/latest/intro.html) library.

Happy learning! 🎉

## 🧪 Instruction for TH1-KP4 

### 1️⃣ Download Data for Null-Tests
To get started with linear and nonlinear beyond-ΛCDM modifications:
- Ensure you are using the following branches:
  - [`cloelib`](https://github.com/cloe-org/cloelib/tree/feature/beyond-lcdm-linear-perturbations): `feature/beyond-lcdm-linear-perturbations`
  - [`MGEmus`](https://github.com/nebblu/MGEmus/tree/mu-sigma-q1): `mu-sigma-q1`
- Navigate to `tutorials/th1-kp4/mgrowth.ipynb` and download the required data from Zenodo executing one of the first cells there:
  - `nz_example.fits`: n(z) distributions for sources and lenses
  - `cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode_copy.npy`: Gaussian covariance
  - `mixmat_identity_5000_binned.fits`: Mixing matrices
  - `synth_cells_5000_binned.fits`: Noiseless synthetic data computed with HMcode2020

### 2️⃣ Run Shear-Only Analysis
#### a. Prepare the Environment
- Copy `EuclidLikelihood_WL_Cls.py` from this directory to `cloelike/cloelike` alongside other `EuclidLikelihood_*.py` files.
- Recompile `cloelike`:
  ```bash
  pip install .
  ```

#### b. Familiarize with the `scripts` Folder
The `scripts` folder is structured as follows:
```
scripts/
│── plotting/  
    │── figs/   
│── sampling/     
    │── chains/
        │── hdf5/  
    │── inifiles/     
    │── scalecuts/  
│── utils/    
```

#### c. Run a Single Likelihood Evaluation
From `scripts/sampling`, execute a single likelihood evaluation for different scenarios (`LCDM_NL`, `MG_L`, `MG_NL`) by modifying the options in the Python script:
```bash
python test_onelike_evaluation.py
```
The output test files will include:
- **Header**: Information on scale cuts, varied parameters, priors, etc.
- **Footer**: Example output:
  ```
  First call (includes JIT compilation)
  loglikelihood = -37.361222
  evaluation took 2.2407 s (--> 0:00:02.240670 hh:mm:ss)
  ##############################################################
  Second call (JIT compiled)
  loglikelihood = -37.361222
  evaluation took 0.0349 s (--> 0:00:00.034887 hh:mm:ss)
  ##############################################################
  Third call (timing verification)
  loglikelihood = -37.361222
  evaluation took 0.0348 s (--> 0:00:00.034773 hh:mm:ss)
  ```

#### d. Use Nautilus for Sampling
Run [`Nautilus`](https://nautilus-sampler.readthedocs.io/en/latest/guides/parallelization.html) using one of the options:
- Shared-memory multiprocessing
- MPI (recommended for clusters like Cuillin)

Refer to `nautilus_example_update.py` and `nautilus_example_update_mpi.py` for detailed explanations.

#### e. Quick Parameter Variation
For a quick run, vary only 2 parameters while fixing others to their fiducial values. Modify the `inifiles\params_model_shear.yaml` accordingly, 
the parameters in the inifiles are defined as follows:
```yaml
Omega_cdm0:
    type: 'U'  # Uniform prior
    p0: 0.27   # Fixed value
    p1: 0.2    # Prior lower limit
    p2: 0.5    # Prior upper limit 

ombh2:
    type: 'G'   # Gaussian prior
    p0: 0.0227  # Fixed value
    p1: 0.0227  # Mean of the prior
    p2: 0.00038 # Std of the prior
```
To ignore baryons in HMCode, leave `p0` empty:
```yaml
log10TAGN: 
    type: 'F'
    p0: 
    p1: 7.6
    p2: 8.3  
```

#### f. Example Slurm File
Check out the example Slurm file: `CLOE_test_mpi.sbatch`.

### 3️⃣ Visualize Results
#### a. Generate Plots
- Navigate to `plotting` and adapt `plot_posterior_minimal.py` to your chains.
- Run the script to generate plots.

#### b. View Saved Figures
- The plots will be saved in `plotting/figs`.

## 🔧 Features
- Tutorials on how to run `cloelib`, `cloelike` and get around the cloe-org ecosystem

## 📂 Structure
The repository is organized as follows:

```
playground/
│── tutorials/       # Jupyter notebooks for cosmological codes, computing observables and evaluating the likelihood with cloelib and cloelike
│── validation/      # Jupyter notebooks for comparison of cosmological observables against `cloelib`
│── exercises/       # Jupyter notebooks with exercises that use `cloelib` for teaching purposes
│── scripts/         # Python scripts with sampling examples
│── README.md        # This file
```

## 📦 Installation
To use this repository, clone it, no installation needed!

```bash
git clone https://github.com/cloe-org/playground.git
cd playground
```

It might require as dependencies `cloelib`, `cloelike`, `euclidlib` and others.

## 🚀 Usage
You can run the provided notebooks for experimentation!

```bash
jupyter notebook tutorials/observables/photo.ipynb
jupyter notebook tutorials/observables/spectro.ipynb
jupyter notebook tutorials/observables/bao.ipynb
```

## 📬 Contact
For any questions or discussions, feel free to open an issue or reach out to the [cloe-maintainers](https://github.com/orgs/cloe-org/teams/cloe-maintainers).

## 🤝 Contributing
This project follows the [all-contributors](https://allcontributors.org) specification. Contributions of any kind welcome!

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#review-gcanasherrera" title="Reviewed Pull Requests">👀</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a> <a href="#example-gcanasherrera" title="Examples">💡</a> <a href="#infra-gcanasherrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-gcanasherrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-gcanasherrera" title="Maintenance">🚧</a> <a href="#projectManagement-gcanasherrera" title="Project Management">📆</a> <a href="#tutorial-gcanasherrera" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chiaramoretti"><img src="https://avatars.githubusercontent.com/u/12472732?v=4?s=100" width="100px;" alt="Chiara Moretti"/><br /><sub><b>Chiara Moretti</b></sub></a><br /><a href="#code-chiaramoretti" title="Code">💻</a> <a href="#review-chiaramoretti" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-chiaramoretti" title="Maintenance">🚧</a> <a href="#tutorial-chiaramoretti" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaPezzotta"><img src="https://avatars.githubusercontent.com/u/29603598?v=4?s=100" width="100px;" alt="AndreaPezzotta"/><br /><sub><b>AndreaPezzotta</b></sub></a><br /><a href="#code-AndreaPezzotta" title="Code">💻</a> <a href="#review-AndreaPezzotta" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-AndreaPezzotta" title="Maintenance">🚧</a> <a href="#tutorial-AndreaPezzotta" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PedroCarrilho"><img src="https://avatars.githubusercontent.com/u/60090062?v=4?s=100" width="100px;" alt="Pedro Carrilho"/><br /><sub><b>Pedro Carrilho</b></sub></a><br /><a href="#code-PedroCarrilho" title="Code">💻</a> <a href="#review-PedroCarrilho" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-PedroCarrilho" title="Maintenance">🚧</a> <a href="#tutorial-PedroCarrilho" title="Tutorials">✅</a> <a href="#ideas-PedroCarrilho" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="#code-santiagocasas" title="Code">💻</a> <a href="#review-santiagocasas" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josecolomanadal"><img src="https://avatars.githubusercontent.com/u/83759085?v=4?s=100" width="100px;" alt="Jose Coloma Nadal"/><br /><sub><b>Jose Coloma Nadal</b></sub></a><br /><a href="#code-josecolomanadal" title="Code">💻</a> <a href="#tutorial-josecolomanadal" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/llinke1"><img src="https://avatars.githubusercontent.com/u/42432333?v=4?s=100" width="100px;" alt="Laila Linke"/><br /><sub><b>Laila Linke</b></sub></a><br /><a href="#code-llinke1" title="Code">💻</a> <a href="#tutorial-llinke1" title="Tutorials">✅</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#code-itutusaus" title="Code">💻</a> <a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

