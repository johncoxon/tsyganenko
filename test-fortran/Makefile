#!/bin/bash
# Makefile for building Tsyganenko and Fortran code to run it

# Declare default compilers
LOCAL_F77 = gfortran
# Use default compilers unless specified by command line input
F77 = $(LOCAL_F77)

# Declare optional and required compilation flags
OPT_FLAGS = -O2 -fbacktrace -fno-automatic -fPIC
REQ_FLAGS = -w

# Declare source code
SOURCES = ../src/geopack_tsyganenko/T96.for \
	  ../src/geopack_tsyganenko/T01_01c.for
SRC = Geopack-2008_2020-01-01.for
SRC_2010 = Geopack-2008_2010-11-30.for
SRC_PY = ../src/geopack_tsyganenko/Geopack.for
TEST_SRC = test_geopack.f90
OBJ = $(SOURCES:.f=.o) $(SRC:.for=.o)

PYF = $(SRC:.for=.pyf)
PYO = $(PYF:.pyf=.so)

# Define operations
all:
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) -c $(SRC) $(SOURCES)
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) $(SRC) $(SOURCES) $(TEST_SRC) -o test_geopack

2010:
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) -c $(SRC_2010) $(SOURCES)
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) $(SRC_2010) $(SOURCES) $(TEST_SRC) -o test_geopack_2010

py:
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) -c $(SRC_PY) $(SOURCES)
	$(F77) $(REQ_FLAGS) $(OPT_FLAGS) $(SRC_PY) $(SOURCES) test_geopack_py2f.f90 -o test_geopack_f2py

clean:
	rm -f $(OBJ) $(PYO)