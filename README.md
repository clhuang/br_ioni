br\_ioni
=======
Framework for emission and spectra simulation of bifrost simulations

Prerequisites
------------
- Python 2.7, with Numpy and Scipy (Anaconda distribution recommended)
- PyCUDA, CUDA 5
- Kivy 1.8.0 (for running GUI)

Ensure that environment variable CUDA_HOME is set correctly.

To run the GUI:
Run "kivy br\_ioni [args]" in the parent directory of the repository.
The parameters are:
{tdi, static}: whether to use time-dependent ionization outputs, or use static ionization equilibrium
--infile: optional first simulation file to use
--tdiparam: optional location of time-dependent ionization parameter file

File outputs of the GUI are documented in br\_ioni.EmissivityRenderer.save\_irender or save\_ilrender.

Input Format
------------
qsmag\_xxxx\_###.aux/idl/snap files should all be placed in one directory, say ~/foo. This folder should also contain eostable.dat, ionization.dat, mesh.dat, tabparam.in, etc.
Within foo, there should be another folder foo/data containing lookup tables for specific elements, e.g. he2\_303.dat or fe9\_184.dat.

When running br\_ioni, simply pick one of the qsmag\_xxxx\_### files and it will load the rest of the files automatically.
