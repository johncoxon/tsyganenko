#!/bin/bash
MODULE := tsygFort

# Makefile for building Tsyganenko and compliling it with f2py

# Declare default compilers
LOCAL_F77  = gfortran
# Use default compilers unless specified by command line input
#echo 'Use F2PY and F77 to specify the fortran-to-python and fortran complilers'
F77  = $(LOCAL_F77)

# Declare optional and required compilation flags
OPT_FLAGS = -O2 -fbacktrace -fno-automatic -fPIC
REQ_FLAGS = -w

# Declare source code
SOURCES = T96.f \
	  T02.f
SRC = Geopack-2008_2020-01-01.for
SRC_2010 = Geopack-2008_2010-11-30.for
OBJ = $(SOURCES:.f=.o) $(SRC:.for=.o)

PYF = $(SRC:.for=.pyf)
PYO = $(PYF:.pyf=.so)

# Define operations
all:
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) -c $(SRC) $(SOURCES)
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) $(SRC) $(SOURCES) test_geopack.f90 -o test_geopack

2010:
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) -c $(SRC_2010) $(SOURCES)
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) $(SRC_2010) $(SOURCES) test_geopack.f90 -o test_geopack

clean:
	rm -f $(OBJ) $(PYO)