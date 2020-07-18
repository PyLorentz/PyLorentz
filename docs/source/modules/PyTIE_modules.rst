.. _PyTIEapi:

PyTIE
=====
The following modules contain the base code for the TIE reconstruction, as well as display functions and the microscope class which is used for simulating LTEM images. 

- ``TIE_reconstruct`` contains the final setup and TIE solver
- ``TIE_params`` is a class that contains the data and reconstruction parameters.
- ``microscopes`` is a class containing all microscope parameters as well as methods for simulating LTEM images. 
- ``TIE_helper`` has loading and display functions
- ``colorwheel`` is for visualizing 2D vector images. 
- ``longitudinal_deriv`` still has bugs, but calculates the intensity derivative through a full stack rather than through the central difference method. 


.. toctree::
   :maxdepth: 4

   TIE_reconstruct
   TIE_params
   microscopes
   TIE_helper
   colorwheel
   longitudinal_deriv
