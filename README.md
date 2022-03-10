# PyLorentz
PyLorentz is a codebase designed for analyzing Lorentz Transmission Electron Microscopy (LTEM) data. There are three primary features and functions: 

- PyTIE -- Reconstructing the magnetic induction from LTEM images using the Transport of Intensity Equation (TIE)
- SimLTEM -- Simulating phase shift and LTEM images from a given magnetization 
- GUI -- GUI provided for PyTIE reconstruction.

For full documentation please check out our documentation pages: [![Documentation Status](https://readthedocs.org/projects/pylorentztem/badge/?version=latest)](https://pylorentztem.readthedocs.io/en/latest/?badge=latest) 

If you use PyLorentz, please cite our [paper](https://doi.org/10.1103/PhysRevApplied.15.044025) [1] and this PyLorentz code: [![DOI](https://zenodo.org/badge/263821805.svg)](https://zenodo.org/badge/latestdoi/263821805)


## Features
### PyTIE 
* Uses inverse Laplacian method to solve the Transport of Intensity Equation (TIE)
* Can reconstruct the magnetization from samples of variable thickness by using two through focal series (tfs) taken with opposite electron beam directions [2]. 
* Samples of uniform thicknss can be reconstructed from a single tfs.
* Thin samples of uniform thickness, from which the only source of contrast is magnetic Fresnel contrast, can be reconstructed with a single defocused image using Single-Image-TIE (SITIE). 

	* This  method does not apply to all samples; for more information please refer to Chess et al. [3]. 

* The TIE and SITIE solvers can implement Tikhonov regularization to remove low-frequency noise [2]. 

	* Results reconstructed with a Tikhonov filter are no longer quantitative, but a Tikhonov filter can greatly increase the range of experimental data that can be reconstructed. 

* Symmetric extensions of the image can be created to reduce Fourier processing edge-effects [4]. 
* Subregions of the images can be selected interactively to improve processing time or remove unwanted regions of large contrast (such as the edge of a TEM window) that would otherwise interfere with reconstruction. 

	* At large aspect ratios, Fourier sampling effects become more pronounced and directionally affect the reconstructed magnetization. Therefore non square images are not quantitative, though symmetrizing the image can greatly reduce this effect.

### SimLTEM
* Easily import .omf and .ovf file outputs from OOMMF and mumax. 
* Calculate electron phase shift through samples of a given magnetization with either the Mansuripur algorithm or linear superposition method. 
* Simulate LTEM images from these phase shifts and reconstruct the magnetic induction for comparison with experimental results. 
* Automate full tilt series for tomographic reconstruction of 3D magnetic samples. 
* PyLorentz code is easily integrated into existing python workflows. 

### GUI
* TIE reconstruction through a graphical user interface (gui) 
* Additional features include improved region selection and easily images before saving. 

## Getting started
With the exception of the gui, this code is intended to be run in Jupyter notebooks and several examples are provided. You can clone the repo directly, fork the project, or download the files directly in a .zip. 


Several standard packages are required which can be installed with conda or pip. Environment.yml files are included in the /envs/ folder. Select the appropriate file for your system and create the environment from a command line with 
```
conda env create -f environment.yml
```
Activate with either 
```
source activate Pylorentz
```
or
```
conda activate Pylorentz
```
depending on operating system before opening a notebook. 

**Example Data**

We recommend running the template files provided in the ``/Examples/`` directory with the provided example data. You can download an experimental dataset from the [Materials Data Facility](https://doi.org/10.18126/z9tc-i8bf), which contains a through focus series (tfs) in both flipped and unflipped orientations as well as an aligned image stack. These files are ready to be used in the ``TIE_template.ipynb``. 

For ``SIM_template.ipynb``, there is an ``example_mumax.ovf`` file which can be used directly. 

## Data organization
If you have both a flip and unflip stack your data should be set up as follows:  

    datafolder/    flip/     -im1.dm3  
                             -im2.dm3  
                                ...  
                             +im1.dm3  
                             +im2.dm3  
                                ...  
                             0im.dm3    
                   unflip/   -im1.dm3  
                             -im2.dm3  
                                 .  
                                 .  
                             +im1.dm3  
                             +im2.dm3  
                                 .  
                                 .  
                              0im.dm3  
                   flsfile.fls 
                   full_align.tif  
  
If your flip and unflip filenames aren't the same you can also have two fls files, just change the optional argument:  ``flip_fls_file = "your flip fls path"``  
  
However if you have just one stack (no flip stack) then your data should be in a folder labeled 'tfs' 

    datafolder/    tfs/      -im1.dm3  
                             -im2.dm3  
                                ...  
                             +im1.dm3  
                             +im2.dm3  
                                ...  
                              0im.dm3    
                   flsfile.fls 
                   full_align.tif  
                   
## References
[1] McCray, A. R. C., Cote, T., Li, Y., Petford-Long, A. K. & Phatak, C. Understanding Complex Magnetic Spin Textures with Simulation-Assisted Lorentz Transmission Electron Microscopy. Phys. Rev. Appl. 15, 044025 (2021).   
[2] Humphrey, E., Phatak, C., Petford-Long, A. K. & De Graef, M. Separation of electrostatic and magnetic phase shifts using a modified transport-of-intensity equation. Ultramicroscopy 139, 5–12 (2014).   
[3] Chess, J. J. et al. Streamlined approach to mapping the magnetic induction of skyrmionic materials. Ultramicroscopy 177, 78–83 (2018).   
[4] Volkov, V. V, Zhu, Y. & Graef, M. De. A New Symmetrized Solution for Phase Retrieval using the TI. Micron 33, 411–416 (2002).
   
## License

This project is licensed under the BSD License - see the [LICENSE.md](https://github.com/PyLorentz/PyLorentz/blob/master/LICENSE) file for details. 
