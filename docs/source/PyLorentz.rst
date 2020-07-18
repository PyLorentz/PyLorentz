PyLorentz Features and Code
===========================
The PyLorentz code is written to be accessible and easily integrated into existing work-flows. However it is still possible to get wrong results if one doesn't have familiarity with the systems involved, which in this case are both the materials involved as well as LTEM. It can be easy to arrive at results that aren't physically possible, especially when simulating samples. PyLorentz can not protect against this beyond providing a few :ref:`references <LTEM_background>` that give some background on LTEM. 

Additionally, while the TIE GUI is a complete product intending to make PyLorentz accessible to everyone, to access all of the features and get the most out of this product a user would find it helpful to be familiar with basic python.  

.. toctree::
   :maxdepth: 2

   ipynb/Getting_started
   API

Features
--------

**PyTIE**

* Uses inverse Laplacian method to solve the Transport of Intensity Equation (TIE)
* Can reconstruct the magnetization from samples of variable thickness by using two through focal series (tfs) taken with opposite electron beam directions. :ref:`[1] <TIE_refs>`
* Samples of uniform thickness can be reconstructed from a single tfs.
* Thin samples of uniform thickness, from which the only source of contrast is magnetic Fresnel contrast, can be reconstructed with a single defocused image using Single-Image-TIE (SITIE). 

	* This  method does not apply to all samples; for more information please refer to Chess et al. :ref:`[2]<TIE_refs>`. 

* The TIE and SITIE solvers can implement Tikhonov regularization to remove low-frequency noise :ref:`[1]<TIE_refs>`. 

	* Results reconstructed with a Tikhonov filter are no longer quantitative, but a Tikhonov filter can greatly increase the range of experimental data that can be reconstructed. 

* Symmetric extensions of the image can be created to reduce Fourier processing edge-effects :ref:`[3]<TIE_refs>`. 
* Subregions of the images can be selected interactively to improve processing time or remove unwanted regions of large contrast (such as the edge of a TEM window) that would otherwise interfere with reconstruction. 

	* At large aspect ratios, Fourier sampling effects become more pronounced and directionally affect the reconstructed magnetization. Therefore non square images are not quantitative, though symmetrizing the image can greatly reduce this effect.

**SimLTEM**

* Easily import .omf and .ovf file outputs from OOMMF and Mumax. 
* Calculate electron phase shift through samples of a given magnetization with either the Mansuripur algorithm or linear superposition method. 
* Simulate LTEM images from these phase shifts and reconstruct the magnetic induction for comparison with experimental results. 
* Automate full tilt series for tomographic reconstruction of 3D magnetic samples. 
* PyLorentz code is easily integrated into existing python work-flows. 

**GUI/Align**

* TIE reconstruction through a graphical user interface (GUI) 
* Additional features include improved region selection and easily images before saving. 
* Image registration routines incorporated (via `FIJI <https://fiji.sc/>`_) for aligning experimental data. 

.. _TIE_refs:

TIE References
--------------

(1) Humphrey, E., Phatak, C., Petford-Long, A. K. & De Graef, M. Separation of electrostatic and magnetic phase shifts using a modified transport-of-intensity equation. Ultramicroscopy 139, 5–12 (2014).   
(2) Chess, J. J. et al. Streamlined approach to mapping the magnetic induction of skyrmionic materials. Ultramicroscopy 177, 78–83 (2018).   
(3) Volkov, V. V, Zhu, Y. & Graef, M. De. A New Symmetrized Solution for Phase Retrieval using the TI. 33, 411–416 (2002).   




