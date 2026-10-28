# The nanobind extension bounded_compositions._core. Built by scikit-build-core
# (pyproject.toml), or by hand with -DBCOMP_BUILD_PYTHON=ON.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC STABLE_ABI src/core.cpp)
# Not bounded_compositions_warnings: nanobind's headers trip -Wshadow.
target_link_libraries(_core PRIVATE bounded_compositions::bounded_compositions)
install(TARGETS _core LIBRARY DESTINATION bounded_compositions)
