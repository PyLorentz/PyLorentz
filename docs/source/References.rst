Further references
======================


.. _LTEM_background:

Linear Superposition Method
----------------------------

For details regarding the linear superposition method of calculating the electron phase shift through a magnetic sample, please see our `2019 paper <https://doi.org/10.1103/PhysRevApplied.15.044025>`_:

        McCray, A. R. C., Cote, T., Li, Y., Petford-Long, A. K. & Phatak, C. Understanding Complex Magnetic Spin Textures with Simulation-Assisted Lorentz Transmission Electron Microscopy. Phys. Rev. Appl. 15, 044025 (2021).

Single Image Phase Reconstruction via Automatic Differentiation (SIPRAD)
--------------------------------------------------------------------------

For further information on how automatic differentiation (AD) can be used for phase reconstruction by applying a generative deep image prior (DIP), see our `2024 paper <https://doi.org/10.1038/s41524-024-01285-8>`_:

        McCray, A. R. C., Zhou, T., Kandel, S. et al. AI-enabled Lorentz microscopy for quantitative imaging of nanoscale magnetic spin textures. npj Comput Mater 10, 111 (2024).

This work was based on an original paper that demonstrated the applicability of AD to phase reconstructions by `Zhou, Cherukara, and Phatak. <https://doi.org/10.1038/s41524-021-00600-x>`_:

        Zhou, T., Cherukara, M. & Phatak, C. Differential programming enabled functional imaging with Lorentz transmission electron microscopy. npj Comput Mater 7, 141 (2021).



LTEM References
---------------

For a brief introduction to Lorentz Transmission Electron Microscopy, the following papers and textbooks may be of some assistance:

- `Phatak, C., Petford-Long, A. K. & De Graef, M. Recent advances in Lorentz microscopy. <https://doi.org/10.1016/j.cossms.2016.01.002>`_

- `De Graef, M. & Zhu, Y. Quantitative noninterferometric Lorentz microscopy. <https://doi.org/10.1063/1.1355337>`_

- `De Graef, M. Introduction to conventional transmission electron microscopy. <https://doi.org/10.1017/CBO9780511615092>`_

- `Reimer, L. & Kohl, H. Transmission Electron Microscopy Physics of Image Formation. <https://doi.org/10.1007/978-0-387-40093-8>`_

- `Williams, D. & Carter, C. Transmission Electron Microscopy. <https://doi.org/10.1007/978-0-387-76501-3>`_


Micromagnetics References
----------------------------------------

The simulation side of PyLorentz begins with the output of micromagnetic simulations. PyLorentz has been tested with `OOMMF <https://math.nist.gov/oommf/>`_ and `MuMax <https://mumax.github.io/>`_; we recommend you refer to their documentation pages for background and information on setting up your own micromagnetic simulations.

There is also `Ubermag <https://ubermag.github.io/>`_, a project that interfaces with both OOMMF and Mumax allowing one to run micromagnetic simulations though Python in Jupyter notebooks. It has helpful display functionalities and makes it easier to begin using micromagnetics without learning the scripting languages.

For those wanting to get started with OOMMF or Ubermag, the Online Spintronics Seminar Series presented a series of `video tutorials <https://www.spintalks.org/tutorials>`_ that are very helpful.
