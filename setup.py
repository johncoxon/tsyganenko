#!/usr/bin/env python
from numpy.distutils.core import setup, Extension

ext = Extension('geopack',
  sources=['tsyganenko/geopack_py.pyf','tsyganenko/geopack_py.for','tsyganenko/T96.f','tsyganenko/T02.f'])

setup (name = "Tsyganenko",
       version = "2020.1",
       description = "A wrapper to call fortran routines from the Tsyganenko models",
       author = "John Coxon and Sebastien de Larquier",
       author_email = "work@johncoxon.co.uk",
       url = "",
       long_description =
        """
For more information on the Tsyganenko gemagnetic field models, go to
http://ccmc.gsfc.nasa.gov/models/modelinfo.php?model=Tsyganenko%20Model
        """,
       packages = ['tsyganenko'],
       ext_modules = [ext],
       keywords=['Scientific/Space'],
       classifiers=[
                   "Programming Language :: Python/Fortran"
                   ]
        )
