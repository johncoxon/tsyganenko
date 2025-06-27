# tsyganenko

[![Zenodo badge](https://zenodo.org/badge/190026596.svg)](https://doi.org/10.5281/zenodo.3937276)
[![Tests badge](https://github.com/johncoxon/tsyganenko/actions/workflows/tests.yaml/badge.svg)](https://github.com/johncoxon/tsyganenko/actions/workflows/tests.yaml)

A Python wrapper for N. A. Tsyganenko’s field-line tracing routines.

[For information on the models and routines that are wrapped by this package, please visit Empirical Magnetosphere Models by N. A. Tsyganenko.](https://geo.phys.spbu.ru/~tsyganenko/empirical-models/)

## Citation

When using this software, please cite [the Zenodo record](https://doi.org/10.5281/zenodo.3937277) as well as following the instructions on [N. A. Tsyganenko's website](https://geo.phys.spbu.ru/~tsyganenko/empirical-models/).

## Copyright

All Fortran files and their scientific contents are developed by and belong to N. A. Tsyganenko and colleagues.

The Python wrappers were originally written by Sebastien de Larquier in 2012 and are now maintained by John C Coxon.

## Funding

John C Coxon was supported during this work by Science and Technology Facilities Council (STFC) Consolidated Grants ST/R000719/1 and ST/V000942/1, and Ernest Rutherford Fellowship ST/V004883/1.

## Installation

    pip install tsyganenko
    pytest tests/tests.py

## Usage

To use this module, simply follow the example provided in the Trace object docstring.

    import tsyganenko as tsy
    tsy.Trace?

Alternatively, there are example notebooks provided which can be used to explore what this module can do. To access these, run:

    cd notebooks
    jupyter notebook