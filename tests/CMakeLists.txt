find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  set(catch2_system "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    set(catch2_system SYSTEM)  # keep our -Werror off Catch2's own headers
  endif()
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.16.0
    GIT_SHALLOW TRUE
    ${catch2_system})
  FetchContent_MakeAvailable(Catch2)
endif()
include(Catch)

add_executable(bounded_compositions_tests
  test_coordinate.cpp
  test_executor.cpp
  test_fresh.cpp
  test_groups.cpp
  test_limbs.cpp
  test_log_counting.cpp
  test_natural.cpp
  test_region.cpp
  test_repair.cpp
  test_sequence.cpp
  test_sequence_vectors.cpp
  test_smoke.cpp
  test_space.cpp
  test_support.cpp)
target_link_libraries(bounded_compositions_tests
  PRIVATE bounded_compositions::bounded_compositions Catch2::Catch2WithMain bounded_compositions_warnings)
# Tests of private pieces (Sobol tables) include headers from src/.
target_include_directories(bounded_compositions_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
catch_discover_tests(bounded_compositions_tests)

# The library rebuilt with a two-limb kernel budget, so the limit is testable
# with small tables.
add_executable(bounded_compositions_limit_tests
  test_limits.cpp
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
target_compile_definitions(bounded_compositions_limit_tests PRIVATE
  BCOMP_MAX_LIMBS=2 BCOMP_VERSION_STRING="${PROJECT_VERSION}")
target_link_libraries(bounded_compositions_limit_tests
  PRIVATE bounded_compositions_kernel Threads::Threads Catch2::Catch2WithMain bounded_compositions_warnings)
catch_discover_tests(bounded_compositions_limit_tests TEST_PREFIX "limit: ")

# Thread start failures, injected by interposing glibc's pthread_create.
# Sanitizers intercept pthread_create themselves, so their builds skip it.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
  add_executable(bounded_compositions_pool_failure_tests test_pool_failure.cpp)
  target_link_libraries(bounded_compositions_pool_failure_tests
    PRIVATE bounded_compositions::bounded_compositions Catch2::Catch2WithMain bounded_compositions_warnings ${CMAKE_DL_LIBS})
  catch_discover_tests(bounded_compositions_pool_failure_tests TEST_PREFIX "pool failure: "
    PROPERTIES TIMEOUT 60)
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME kernel_rules
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/check_kernel.py
            ${PROJECT_SOURCE_DIR}/include/bounded_compositions/kernel)
endif()

# Fuzz properties: a seeded smoke run everywhere, real libFuzzer with Clang.
option(BCOMP_FUZZ "Build the libFuzzer target (Clang only)" OFF)
if(BCOMP_FUZZ)
  add_executable(bounded_compositions_fuzz fuzz/fuzz_properties.cpp)
  target_compile_options(bounded_compositions_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(bounded_compositions_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(bounded_compositions_fuzz PRIVATE bounded_compositions::bounded_compositions bounded_compositions_warnings)
else()
  add_executable(bounded_compositions_fuzz_smoke fuzz/fuzz_properties.cpp fuzz/fuzz_driver.cpp)
  target_link_libraries(bounded_compositions_fuzz_smoke PRIVATE bounded_compositions::bounded_compositions bounded_compositions_warnings)
  add_test(NAME fuzz_smoke COMMAND bounded_compositions_fuzz_smoke)
endif()
