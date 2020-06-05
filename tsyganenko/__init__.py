"""
tsyganenko : a module to trace magnetic field lines using the Tsyganenko models

This package was initially written by Sebastien de Larquier (Virginia Tech).
In 2020, the package was updated by John Coxon (University of Southampton) to
add support for the latest release of geopack-2008.for and Python 3 support.

Copyright (C) 2012 VT SuperDARN Lab

.. moduleauthor:: John Coxon
"""
import geopack
from .trace import *


# Declare the same Earth radius as used in Tsyganenko models (km)
RE = 6371.2
