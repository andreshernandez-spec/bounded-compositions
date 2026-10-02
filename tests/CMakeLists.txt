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
  test_limbs.cpp
  test_natural.cpp
  test_smoke.cpp
  test_space.cpp)
target_link_libraries(bounded_compositions_tests
  PRIVATE bounded_compositions::bounded_compositions Catch2::Catch2WithMain bounded_compositions_warnings)
catch_discover_tests(bounded_compositions_tests)

# The library rebuilt with a two-limb kernel budget, so the limit is testable
# with small tables.
add_executable(bounded_compositions_limit_tests
  test_limits.cpp
  ${PROJECT_SOURCE_DIR}/src/natural.cpp
  ${PROJECT_SOURCE_DIR}/src/space.cpp
  ${PROJECT_SOURCE_DIR}/src/version.cpp)
target_compile_definitions(bounded_compositions_limit_tests PRIVATE
  BCOMP_MAX_LIMBS=2 BCOMP_VERSION_STRING="${PROJECT_VERSION}")
target_link_libraries(bounded_compositions_limit_tests
  PRIVATE bounded_compositions_kernel Threads::Threads Catch2::Catch2WithMain bounded_compositions_warnings)
catch_discover_tests(bounded_compositions_limit_tests TEST_PREFIX "limit: ")

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME kernel_rules
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/check_kernel.py
            ${PROJECT_SOURCE_DIR}/include/bounded_compositions/kernel)
endif()
