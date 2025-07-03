# tsyganenko

[![Zenodo badge](https://zenodo.org/badge/190026596.svg)](https://doi.org/10.5281/zenodo.3937276)
[![Tests badge](https://github.com/johncoxon/tsyganenko/actions/workflows/tests.yaml/badge.svg)](https://github.com/johncoxon/tsyganenko/actions/workflows/tests.yaml)

A Python wrapper for N. A. Tsyganenko’s field-line tracing routines.

[For information on the models and routines that are wrapped by this package, please visit Empirical Magnetosphere
Models by N. A. Tsyganenko.](https://geo.phys.spbu.ru/~tsyganenko/empirical-models/)

## Citation

When using this software, please cite [the Zenodo record](https://zenodo.org/records/15763019) as well as citing the relevant papers.

## Copyright

Geopack and the other Fortran code in this repository are developed by N A Tsyganenko et al. and licenced under the GPL
v3 or later.

The Python wrappers were originally written by Sebastien de Larquier in 2012. They are now maintained by John C Coxon.
Small edits have been made to `Geopack-2008.for` to allow for `f2py` compilation. Code in this repository other than the
original Fortran is licenced under the MIT licence.

## Funding

John C Coxon was supported during this work by Science and Technology Facilities Council (STFC) Consolidated Grants 
ST/R000719/1 and ST/V000942/1, and Ernest Rutherford Fellowship ST/V004883/1.

## Installation

    pip install tsyganenko
    pytest tests/test.py

Please note that at this time, Windows is not well-supported by this package; if you would like to help resolve this,
please check out [issue #15](https://github.com/johncoxon/tsyganenko/issues/15).

Tests take around 1 hour to run, so don’t be alarmed if you see this behaviour on your machine.

The tests are run against the `csv` files in the repository which were created on an Apple Silicon (M1) Mac. Compiling 
the package on other machines leads to disparities which reflect the change in underlying architecture rather than
reflecting changes in code. As such, by default, `tests.py` is extremely permissive; it will test to a relative accuracy
of 0.5 or an absolute accuracy of 10.0. If you want to run tests with the `pytest` defaults, run
`pytest tests/test_exact.py`, and expect a large number of tests to fail unless you are running on an M1 chip.

## Usage

To use this module, simply follow the example provided in the Trace object docstring.

    from tsyganenko.trace import Trace
    help(Trace)

Alternatively, there are example notebooks provided which can be used to explore what this module can do:

    cd notebooks
    jupyter notebook