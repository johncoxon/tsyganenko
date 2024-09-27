cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

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

add_custom_command(
  OUTPUT geopack_pymodule.c geopack_py-f2pywrappers.f
  DEPENDS src/tsyganenko/geopack_py.f
  VERBATIM
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
          "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack_py.f" -m geopack_py --lower)

python_add_library(
  geopack_py MODULE "${CMAKE_CURRENT_BINARY_DIR}/geopack_pymodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/geopack_py-f2pywrappers.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack_py.f" WITH_SOABI)
target_link_libraries(geopack_py PRIVATE fortranobject)

install(TARGETS geopack_py DESTINATION .)