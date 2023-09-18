from numpy.distutils.core import Extension
import setuptools

ext = Extension('geopack',
                sources=['src/tsyganenko/geopack_py.pyf',
                         'src/tsyganenko/geopack_py.for',
                         'src/tsyganenko/T96.f',
                         'src/tsyganenko/T02.f'])

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(author="John Coxon and Sebastien de Larquier",
                 author_email="work@johncoxon.co.uk",
                 classifiers=[
                     "Development Status :: 4 - Beta",
                     "Intended Audience :: Science/Research",
                     "Natural Language :: English",
                     "Programming Language :: Python/Fortran"
                 ],
                 description="A wrapper to call GEOPACK routines in Python.",
                 ext_modules=[ext],
                 install_requires=[
                     "numpy",
                     "matplotlib",
                     "pandas"
                 ],
                 keywords=['Scientific/Space'],
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 name="Tsyganenko",
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 python_requires=">=3.9",
                 url="https://github.com/johncoxon/tsyganenko",
                 version="2020.1",
                 )
