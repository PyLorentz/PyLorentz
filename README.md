:information_source: PyLorentz has had a major update :information_source: We anticipate that this will break nearly all workflows, but it also brings new features and a very important refactor of the base code. Please contact the dev team or raise an issue if you find any bugs. 


# PyLorentz
PyLorentz is a codebase designed for analyzing Lorentz Transmission Electron Microscopy (LTEM) data. There are two primary features and functions: 

- Phase reconstruction: Reconstructing the electron phase shift from LTEM images using either the transport of intensity equation (TIE) or automatic differentiation (AD) based methods
- SimLTEM: Simulating the electron phase shift and LTEM images from a given magnetization configuration, such as that created by micromagnetic simulations.


For full documentation please check out our documentation pages: [![Documentation Status](https://readthedocs.org/projects/pylorentztem/badge/?version=latest)](https://pylorentztem.readthedocs.io/en/latest/?badge=latest) 
# Features

### Phase Reconstruction with TIE

* We use the inverse Laplacian method to solve the Transport of Intensity Equation (TIE)
* This can reconstruct and isolate the magnetic component of the electron phase shift from samples of variable thickness by using two through focal series (tfs) taken with opposite electron beam directions. 
* It can also reconstruct the total electron phase shift from a single tfs. For samples that induce a uniform electrostatic phase shift, this is functionally equivalent to the magnetic phase shift.
* For thin samples of uniform thickness which have only magnetic Fresnel contrast, the magnetic phase shift can be reconstructed from a single defocused image using single-image TIE (SITIE).

### Phase Reconstruction with AD

* We apply automatic differentiation (AD) to reconstruct the electron phase shift with higher spatial and phase resolution than the TIE method.
* We use a generative deep image prior (DIP) which provides excellent robustness to noise and enables phase reconstruction from a single image (SIPRAD). 
* In some cases, this allows us to isolate the magnetic component of the electron phase shift from a non-uniform electrostatic phase shift given a single LTEM image.

### LTEM Image Simulation

* Calculate electron phase shifts through a sample with an arbitrary magnetization, e.g. from a micromagnetics simulation, using either the Mansuripur algorithm or linear superposition method.
* Simulate LTEM images from these phase shifts and reconstruct the magnetic induction for comparison with experimental results.


# Getting Started

PyLorentz is not (yet) hosted on PyPI or conda-forge, and we therefore recommend performing a local install in a virtual environment. First, either clone or download the PyLorentz code from GitHub, and then in a Conda terminal: 
```
conda create -n pylorentz
conda activate pylorentz
```
Navigate to the PyLorentz code folder, and then perform the local install by running: 
```
pip install .
```

#### GPU installation
If you have a GPU and want to use the AD-enabled phase retrival or accelerate the phase shift calculation, PyTorch and Cupy must be installed in the environment as well. We recommend first installing PyTorch according to their [getting started guidelines](https://pytorch.org/get-started/locally/), then installing [CuPy](https://docs.cupy.dev/en/stable/install.html), and then installing PyLorentz locally as suggested above. 


## Demos and example data

We recommend running the demonstration notebooks provided in the ``/Examples/`` directory, which are designed to run with example data that can be [downloaded from Zenodo](https://zenodo.org/records/13147848). This includes both experimental and simulated data for phase reconstructions and LTEM image simulations. 

# Citing

If you use PyLorentz in your work, please cite our relevant paper(s). If you used PyLorentz for LTEM image simulations or TIE/SITIE reconstructions, please cite our [2021 paper](https://doi.org/10.1103/PhysRevApplied.15.044025). If you used SIPRAD or AD based phase reconstructions, please cite our [2024 paper](https://doi.org/10.1038/s41524-024-01285-8). 

@article{PhysRevApplied.15.044025,  
  title = {Understanding Complex Magnetic Spin Textures with Simulation-Assisted Lorentz Transmission Electron Microscopy},  
  author = {McCray, Arthur R.C. and Cote, Timothy and Li, Yue and Petford-Long, Amanda K. and Phatak, Charudatta},  
  journal = {Phys. Rev. Appl.},  
  volume = {15},  
  issue = {4},  
  pages = {044025},  
  numpages = {12},  
  year = {2021},
  publisher = {American Physical Society},  
  doi = {10.1103/PhysRevApplied.15.044025},  
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.15.044025 }  
}  


@article{npjCompMat.10.111,  
  title={AI-enabled Lorentz microscopy for quantitative imaging of nanoscale magnetic spin textures},  
  author={McCray, Arthur RC and Zhou, Tao and Kandel, Saugat and Petford-Long, Amanda and Cherukara, Mathew J and Phatak, Charudatta},  
  journal={npj Computational Materials},  
  volume={10},  
  number={1},  
  pages={111},  
  year={2024},  
  publisher={Nature Publishing Group UK London},  
  doi = {10.1038/s41524-024-01285-8},  
  url = {https://www.nature.com/articles/s41524-024-01285-8 }  
}

# License

This project is licensed under the BSD License - see the [LICENSE.md](https://github.com/PyLorentz/PyLorentz/blob/master/LICENSE) file for details. 
