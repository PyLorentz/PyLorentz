PyLorentz features and code
===========================
The PyLorentz code is written to be accessible and easily integrated into existing work-flows. However it is still possible to get wrong results if one doesn't have familiarity with the systems involved, which in this case are both the sample materials and LTEM as a technique. It can be easy to arrive at results that aren't physically possible, especially when simulating samples. PyLorentz can not protect against this beyond providing a few :ref:`references <LTEM_background>` that give some background on LTEM.

.. toctree::
   :maxdepth: 2

   API

Features
--------

**Phase Reconstruction with TIE**

* We use the inverse Laplacian method to solve the Transport of Intensity Equation (TIE)
* This can reconstruct and isolate the magnetic component of the electron phase shift from samples of variable thickness by using two through focal series (tfs) taken with opposite electron beam directions. :ref:`[1] <refs>`
* It can also reconstruct the total electron phase shift from a single tfs. For samples that induce a uniform electrostatic phase shift, this is functionally equivalent to the magnetic phase shift.
* For thin samples of uniform thickness which have only magnetic Fresnel contrast, the magnetic phase shift can be reconstructed from a single defocused image using single-image TIE (SITIE).

	* This  method does not apply to all samples; for more information please refer to Chess et al. :ref:`[2]<refs>`.

* The TIE and SITIE solvers can implement Tikhonov regularization to remove low-frequency noise :ref:`[1]<refs>`.

	* Results reconstructed with a Tikhonov filter are (in most cases) no longer quantitative, but a Tikhonov filter can greatly increase the range of experimental data that can be reconstructed.

* Symmetric extensions of the image can be created to reduce Fourier processing edge-effects :ref:`[3]<refs>`.
* Subregions of the images can be selected interactively to improve processing time or remove unwanted regions of large contrast (such as the edge of a TEM window) that would otherwise interfere with reconstruction.

	* At large aspect ratios, Fourier sampling effects become more pronounced and directionally affect the reconstructed phase shift. Therefore, non-square images are technically not quantitative, though this is not noticable until high aspect ratios and symmetrizing the image can greatly reduce this effect.

**Phase Reconstruction with AD**

* We apply automatic differentiation (AD) to reconstruct the electron phase shift with higher spatial and phase resolution than the TIE method. :ref:`[4]<refs>`
* We use a generative deep image prior (DIP) which provides excellent robustness to noise and enables phase reconstruction from a single image (SIPRAD). :ref:`[5]<refs>`
* In some cases, this allows us to isolate the magnetic component of the electron phase shift from a non-uniform electrostatic phase shift given a single LTEM image. :ref:`[5]<refs>`

**SimLTEM**

* Easily import .omf and .ovf file outputs from OOMMF and Mumax.
* Calculate electron phase shift through samples of a given magnetization with either the Mansuripur algorithm or linear superposition method. :ref:`[6]<refs>`
* Simulate LTEM images from these phase shifts and reconstruct the magnetic induction for comparison with experimental results.
* PyLorentz code is easily integrated into existing python work-flows.

.. _refs:

References
--------------

(1) Humphrey, E., Phatak, C., Petford-Long, A. K. & De Graef, M. Separation of electrostatic and magnetic phase shifts using a modified transport-of-intensity equation. Ultramicroscopy 139, 5–12 (2014).
(2) Chess, J. J. et al. Streamlined approach to mapping the magnetic induction of skyrmionic materials. Ultramicroscopy 177, 78–83 (2018).
(3) Volkov, V. V, Zhu, Y. & Graef, M. De. A New Symmetrized Solution for Phase Retrieval using the TI. 33, 411–416 (2002).
(4) Zhou, T., Cherukara, M. & Phatak, C. Differential programming enabled functional imaging with Lorentz transmission electron microscopy. npj Comput Mater 7, 141 (2021).
(5) McCray, A. R. C., Zhou, T., Kandel, S. et al. AI-enabled Lorentz microscopy for quantitative imaging of nanoscale magnetic spin textures. npj Comput Mater 10, 111 (2024).
(6) McCray, A. R. C., Cote, T., Li, Y., Petford-Long, A. K. & Phatak, C. Understanding Complex Magnetic Spin Textures with Simulation-Assisted Lorentz Transmission Electron Microscopy. Phys. Rev. Appl. 15, 044025 (2021).

