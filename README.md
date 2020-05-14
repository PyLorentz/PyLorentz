# PyLorentz
Python code for managing Lorentz Transmission Electron Microscopy (LTEM) data. 
There will are three primary components: 
- PyTIE  -- Reconstructing the magnetic induction from LTEM images using the Transport of Intensity Equation (TIE)
- SimLTEM -- Simulating phase shift and LTEM images from a given magnetization 
- Align -- Aligning raw LTEM images so they can be reconstructed
- GUI -- In progress, a GUI for aligning and reconstructing data all in one place. 

PyLorentz has the following DOI (through Zenodo):  
[![DOI](https://zenodo.org/badge/259449841.svg)](https://zenodo.org/badge/latestdoi/259449841)


## Getting started
This code is intended to be run in Jupyter notebooks, with several examples already included. You can clone the repo directly (is this bad practice??), fork the project, or download the files directly in a .zip. 


Several standard packages are required which can be installed with conda or pip. Environment.yml files are included in the /envs/ folder, and more will be added soon. Create the environment with 
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
depending on operating system before opening a notebook
```
Jupyter notebook
```

### Running with example data
You can download an example dataset from: https://doi.org/10.18126/z9tc-i8bf, which contains a full through focus series (tfs) with the sample in both flipped and unflipped orientations, and an aligned stack of the images as well. 

## Data organization
If you have both a flip and unflip stack your data should be set up:  

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
  
If your flip and unflip filenames aren't the same you can also have two fls files, just change the optional argument:  flip_fls_file = "your flip fls path"  
  
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
                   


## License

This project is licensed under the BSD License - see the [LICENSE.md](https://github.com/PyLorentz/PyLorentz/blob/master/LICENSE) file for details. 
