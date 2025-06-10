cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

# F2PY headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

# Define variables
set(FORTRAN_PYF_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack.pyf)
set(FORTRAN_SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack.f)
set(FORTRAN_T96_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/T96.f)
set(FORTRAN_T02_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/T02.f)
set(MODULE_NAME geopack_tsyganenko)
set(F2PY_MODULE ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}module.c)
set(F2PY_WRAPPERS ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}-f2pywrappers.f)

add_custom_command(
  OUTPUT ${F2PY_MODULE} ${F2PY_WRAPPERS}
  DEPENDS
    ${FORTRAN_PYF_FILE}
    ${FORTRAN_SRC_FILE}
    ${FORTRAN_T96_FILE}
    ${FORTRAN_T02_FILE}
  VERBATIM
  COMMAND "${PYTHON_EXECUTABLE}" -m numpy.f2py
    "${FORTRAN_PYF_FILE"}
    "${FORTRAN_SRC_FILE"}
    "${FORTRAN_T96_FILE"}
    "${FORTRAN_T02_FILE"}
    -m ${MODULE_NAME}
    --lower
)

python_add_library(
  geopack_tsyganenko MODULE "${F2PY_MODULE}"
  "${F2PY_WRAPPERS}"
  "${FORTRAN_PYF_FILE}"
  "${FORTRAN_SRC_FILE}"
  "${FORTRAN_T96_FILE}"
  "${FORTRAN_T02_FILE}"
  WITH_SOABI)
target_link_libraries(${MODULE_NAME} PRIVATE fortranobject)

install(TARGETS ${MODULE_NAME} DESTINATION .)