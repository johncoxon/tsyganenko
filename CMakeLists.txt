cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_Fortran_FLAGS "-w")

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
set(F2PY_SO ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}${CMAKE_SHARED_MODULE_SUFFIX})

add_custom_command(
  OUTPUT ${F2PY_SO}
  DEPENDS
    ${FORTRAN_PYF_FILE}
    ${FORTRAN_SRC_FILE}
    ${FORTRAN_T96_FILE}
    ${FORTRAN_T02_FILE}
  VERBATIM
  COMMAND "${PYTHON_EXECUTABLE}" -m numpy.f2py -c
    "${FORTRAN_PYF_FILE}"
    "${FORTRAN_SRC_FILE}"
    "${FORTRAN_T96_FILE}"
    "${FORTRAN_T02_FILE}"
    -m ${MODULE_NAME}
    --lower
)

add_custom_target(${MODULE_NAME}_f2py ALL
  DEPENDS ${F2PY_SO}
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION .
        FILES_MATCHING PATTERN "${MODULE_NAME}*${CMAKE_SHARED_MODULE_SUFFIX}")