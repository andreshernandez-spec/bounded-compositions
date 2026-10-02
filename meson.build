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
  'src/natural.cpp',
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
      'tests/test_limbs.cpp',
      'tests/test_natural.cpp',
      'tests/test_smoke.cpp',
      'tests/test_space.cpp',
      dependencies: [bounded_compositions_dep, catch2])
    test('bounded_compositions_tests', tests, timeout: 300)
    limit_tests = executable('bounded_compositions_limit_tests',
      'tests/test_limits.cpp', 'src/natural.cpp', 'src/space.cpp', 'src/version.cpp',
      include_directories: inc,
      cpp_args: ['-DBCOMP_MAX_LIMBS=2',
                 '-DBCOMP_VERSION_STRING="' + meson.project_version() + '"'],
      dependencies: [threads, catch2])
    test('bounded_compositions_limit_tests', limit_tests)
  endif
endif
