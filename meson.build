project('bounded-compositions', 'cpp',
  version: '0.0.1',
  license: 'BSD-3-Clause',
  meson_version: '>=0.64',
  default_options: ['cpp_std=c++17', 'warning_level=3'])

inc = include_directories('include')
threads = dependency('threads')

# Header-only, device-ready kernel layer.
bounded_compositions_kernel_dep = declare_dependency(include_directories: inc)

bounded_compositions_lib = library('bounded_compositions',
  'src/codec.cpp',
  'src/executor.cpp',
  'src/fresh.cpp',
  'src/natural.cpp',
  'src/region.cpp',
  'src/sequence.cpp',
  'src/sobol.cpp',
  'src/sobol_directions.cpp',
  'src/space.cpp',
  'src/version.cpp',
  include_directories: inc,
  dependencies: threads,
  cpp_args: ['-DBCOMP_VERSION_STRING="' + meson.project_version() + '"'],
  install: not meson.is_subproject())

# What consumers (for example a project vendoring this as a subproject) use.
bounded_compositions_dep = declare_dependency(
  include_directories: inc,
  link_with: bounded_compositions_lib,
  dependencies: threads)
meson.override_dependency('bounded-compositions', bounded_compositions_dep)

if not meson.is_subproject()
  catch2 = dependency('catch2-with-main', version: '>=3', required: false)
  if not catch2.found()
    # Some Catch2 packages (conda-forge) ship CMake config files only.
    catch2 = dependency('Catch2', method: 'cmake', modules: ['Catch2::Catch2WithMain'],
      version: '>=3', required: false)
  endif
  if catch2.found()
    tests = executable('bounded_compositions_tests',
      'tests/test_coordinate.cpp',
      'tests/test_executor.cpp',
      'tests/test_fresh.cpp',
      'tests/test_groups.cpp',
      'tests/test_limbs.cpp',
      'tests/test_log_counting.cpp',
      'tests/test_natural.cpp',
      'tests/test_region.cpp',
      'tests/test_sequence.cpp',
      'tests/test_smoke.cpp',
      'tests/test_space.cpp',
      'tests/test_support.cpp',
      include_directories: include_directories('src'),
      dependencies: [bounded_compositions_dep, catch2])
    test('bounded_compositions_tests', tests, timeout: 300)
    limit_tests = executable('bounded_compositions_limit_tests',
      'tests/test_limits.cpp', 'src/codec.cpp', 'src/executor.cpp', 'src/fresh.cpp', 'src/natural.cpp',
      'src/region.cpp',
      'src/sequence.cpp', 'src/sobol.cpp', 'src/sobol_directions.cpp',
      'src/space.cpp', 'src/version.cpp',
      include_directories: inc,
      cpp_args: ['-DBCOMP_MAX_LIMBS=2',
                 '-DBCOMP_VERSION_STRING="' + meson.project_version() + '"'],
      dependencies: [threads, catch2])
    test('bounded_compositions_limit_tests', limit_tests)
    fuzz_smoke = executable('bounded_compositions_fuzz_smoke',
      'tests/fuzz/fuzz_properties.cpp', 'tests/fuzz/fuzz_driver.cpp',
      dependencies: bounded_compositions_dep)
    test('fuzz_smoke', fuzz_smoke, timeout: 300)
  endif
endif
