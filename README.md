## Licence

Though edits had to be made to the fortran code to accomodate `f2py` compilation, all Fortran files and their scientific contents are developed by and belong to N. A. Tsyganenko and colleagues.

The Python wrappers were originally written by Sebastien de Larquier in 2012 and expanded by John Coxon in 2020. These wrappers allow for the Fortran subroutines to be easily called in Python.

## Installation

To install the Tsyganenko python module, from this directory run:

    cd tsyganenko
    make clean
    make
    cd ..
    python setup.py install

To run the unit tests to confirm that the module behaves as expected, run:

    cd tests
    python -m unittest test.Trace1965to2015

## Use

To use this module, simply follow the example provided in the Trace object docstring.

    import tsyganenko as tsy
    tsy.Trace?

Alternatively, there are example notebooks provided which can be used to explore what this module can do. To access these, run:

    cd notebooks
    jupyter notebook