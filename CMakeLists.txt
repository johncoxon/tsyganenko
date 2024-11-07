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
  OUTPUT geopack_tsyganenkomodule.c geopack_tsyganenko-f2pywrappers.f
  DEPENDS
    src/tsyganenko/geopack.pyf
    src/tsyganenko/geopack.f
    src/tsyganenko/T96.f
    src/tsyganenko/T02.f
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py
    -m geopack_tsyganenko
    --lower
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack.pyf"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/T96.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/T02.f"
)

python_add_library(
  geopack_tsyganenko MODULE "${CMAKE_CURRENT_BINARY_DIR}/geopack_tsyganenkomodule.c"
  "${CMAKE_CURRENT_BINARY_DIR}/geopack_tsyganenko-f2pywrappers.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/geopack.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/T96.f"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/tsyganenko/T02.f"
  WITH_SOABI)
target_link_libraries(geopack_tsyganenko PRIVATE fortranobject)

install(TARGETS geopack_tsyganenko DESTINATION .)