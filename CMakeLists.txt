cmake_minimum_required(VERSION 3.21)

project(bounded-compositions VERSION 0.0.1 LANGUAGES CXX)

option(BCOMP_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(BCOMP_WERROR "Treat warnings as errors" OFF)
option(BCOMP_BUILD_PYTHON "Build the Python extension (python/)" OFF)

if(BCOMP_BUILD_PYTHON)
  # The static library is linked into a shared extension module.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

find_package(Threads REQUIRED)

# Warning flags for code in this repository only, never propagated to users.
add_library(bounded_compositions_warnings INTERFACE)
if(MSVC)
  target_compile_options(bounded_compositions_warnings INTERFACE /W4 /permissive-)
  if(BCOMP_WERROR)
    target_compile_options(bounded_compositions_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(bounded_compositions_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow)
  if(BCOMP_WERROR)
    target_compile_options(bounded_compositions_warnings INTERFACE -Werror)
  endif()
endif()

# Kernel layer: header-only, device-ready (docs/kernel-rules.md).
add_library(bounded_compositions_kernel INTERFACE)
add_library(bounded_compositions::kernel ALIAS bounded_compositions_kernel)
target_include_directories(bounded_compositions_kernel INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(bounded_compositions_kernel INTERFACE cxx_std_17)

# Host core and public API. The tests rebuild these sources with other
# settings (a smaller limb budget, fuzzing coverage).
set(BCOMP_SOURCES
  ${PROJECT_SOURCE_DIR}/src/codec.cpp
  ${PROJECT_SOURCE_DIR}/src/executor.cpp
  ${PROJECT_SOURCE_DIR}/src/fresh.cpp
  ${PROJECT_SOURCE_DIR}/src/natural.cpp
  ${PROJECT_SOURCE_DIR}/src/region.cpp
  ${PROJECT_SOURCE_DIR}/src/repair.cpp
  ${PROJECT_SOURCE_DIR}/src/sequence.cpp
  ${PROJECT_SOURCE_DIR}/src/sobol.cpp
  ${PROJECT_SOURCE_DIR}/src/sobol_directions.cpp
  ${PROJECT_SOURCE_DIR}/src/space.cpp
  ${PROJECT_SOURCE_DIR}/src/version.cpp)
add_library(bounded_compositions ${BCOMP_SOURCES})
add_library(bounded_compositions::bounded_compositions ALIAS bounded_compositions)
target_link_libraries(bounded_compositions
  PUBLIC bounded_compositions_kernel Threads::Threads
  PRIVATE bounded_compositions_warnings)
target_compile_definitions(bounded_compositions PRIVATE
  BCOMP_VERSION_STRING="${PROJECT_VERSION}")
set_target_properties(bounded_compositions PROPERTIES CXX_EXTENSIONS OFF)

# Compile every kernel header on its own with exceptions and RTTI off, so a
# header that needs anything outside the kernel rules fails the build.
file(GLOB BCOMP_KERNEL_HEADERS CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/include/bounded_compositions/kernel/*.hpp)
set(BCOMP_KERNEL_TUS "")
foreach(header IN LISTS BCOMP_KERNEL_HEADERS)
  get_filename_component(name ${header} NAME_WE)
  set(tu ${PROJECT_BINARY_DIR}/kernel_standalone/${name}.cpp)
  file(CONFIGURE OUTPUT ${tu}
    CONTENT "#include \"bounded_compositions/kernel/${name}.hpp\"\n")
  list(APPEND BCOMP_KERNEL_TUS ${tu})
endforeach()
add_library(bounded_compositions_kernel_standalone OBJECT ${BCOMP_KERNEL_TUS})
target_link_libraries(bounded_compositions_kernel_standalone
  PRIVATE bounded_compositions_kernel bounded_compositions_warnings)
if(MSVC)
  target_compile_options(bounded_compositions_kernel_standalone PRIVATE /GR- /EHs-c-)
  target_compile_definitions(bounded_compositions_kernel_standalone PRIVATE _HAS_EXCEPTIONS=0)
else()
  target_compile_options(bounded_compositions_kernel_standalone PRIVATE -fno-exceptions -fno-rtti)
endif()

if(BCOMP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

option(BCOMP_BUILD_BENCH "Build the benchmarks" OFF)
if(BCOMP_BUILD_BENCH)
  add_executable(bounded_compositions_unrank_bench bench/unrank_bench.cpp)
  add_executable(bounded_compositions_sampling_report bench/sampling_report.cpp)
  target_link_libraries(bounded_compositions_sampling_report
    PRIVATE bounded_compositions::bounded_compositions bounded_compositions_warnings)
  target_link_libraries(bounded_compositions_unrank_bench PRIVATE bounded_compositions::bounded_compositions bounded_compositions_warnings)
endif()

if(BCOMP_BUILD_PYTHON)
  add_subdirectory(python)
endif()
