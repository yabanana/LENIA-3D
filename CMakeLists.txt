cmake_minimum_required(VERSION 3.20)
project(LeniaThesis VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LENIA_BUILD_VIZ "Build visualization (requires raylib)" ON)
option(LENIA_BUILD_TESTS "Build test suite" ON)

# FFTW3 single-precision (fftw3f) — installed system-wide
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
# fftw3f_threads may not have a separate .pc file (e.g. on Arch Linux),
# so we try pkg-config first and fall back to manual linking.
pkg_check_modules(FFTW3F_THREADS QUIET fftw3f_threads)
if(NOT FFTW3F_THREADS_FOUND)
    set(FFTW3F_THREADS_LIBRARIES fftw3f_threads)
    set(FFTW3F_THREADS_INCLUDE_DIRS ${FFTW3F_INCLUDE_DIRS})
endif()

# OpenMP
find_package(OpenMP REQUIRED)

include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# raylib via FetchContent
if(LENIA_BUILD_VIZ)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 4.0.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(raylib)
endif()

# GoogleTest
if(LENIA_BUILD_TESTS)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# --- Core library (no graphics dependencies) ---
add_library(lenia_core STATIC
    src/core/grid.cc
    src/core/fft_engine.cc
    src/core/kernel.cc
    src/core/growth.cc
    src/core/lenia.cc
    src/core/config.cc
)
target_include_directories(lenia_core PUBLIC src)

# Enable -ffast-math on growth.cc to allow SIMD vectorization of exp().
# Error is negligible for Lenia dynamics (states in [0,1]).
set_source_files_properties(src/core/growth.cc PROPERTIES
    COMPILE_OPTIONS "-ffast-math"
)
target_link_libraries(lenia_core PUBLIC
    ${FFTW3F_LIBRARIES}
    ${FFTW3F_THREADS_LIBRARIES}
    OpenMP::OpenMP_CXX
    m
)
target_include_directories(lenia_core PUBLIC ${FFTW3F_INCLUDE_DIRS})

# --- I/O library ---
add_library(lenia_io STATIC
    src/io/csv_exporter.cc
    src/io/benchmark.cc
)
target_include_directories(lenia_io PUBLIC src)
target_link_libraries(lenia_io PUBLIC lenia_core)

# --- Visualization ---
if(LENIA_BUILD_VIZ)
    add_library(lenia_viz STATIC
        src/viz/renderer.cc
        src/viz/colormap.cc
        src/viz/marching_cubes.cc
        src/viz/camera_ctrl.cc
        src/viz/gui_overlay.cc
        src/viz/slice_view.cc
        src/io/screenshot.cc
    )
    target_include_directories(lenia_viz PUBLIC src)
    target_include_directories(lenia_viz PUBLIC ${CMAKE_SOURCE_DIR}/extern/raygui)
    target_link_libraries(lenia_viz PUBLIC lenia_core lenia_io raylib)

    # Main executable with visualization
    add_executable(lenia_viz_app src/main.cc)
    target_link_libraries(lenia_viz_app PRIVATE lenia_viz lenia_io lenia_core)
    set_target_properties(lenia_viz_app PROPERTIES OUTPUT_NAME lenia_viz)
endif()

# Headless benchmark executable
add_executable(lenia_bench src/main_bench.cc)
target_link_libraries(lenia_bench PRIVATE lenia_io lenia_core)

# Headless parameter sweep
add_executable(lenia_sweep src/main_sweep.cc)
target_link_libraries(lenia_sweep PRIVATE lenia_core)

# Search library (3D metrics + classification)
add_library(lenia_search_lib STATIC src/search/metrics_3d.cc)
target_include_directories(lenia_search_lib PUBLIC src)
target_link_libraries(lenia_search_lib PUBLIC lenia_core)

# Automated parameter-space search tool
add_executable(lenia_search src/main_search.cc)
target_link_libraries(lenia_search PRIVATE lenia_search_lib lenia_core)

# --- Tests ---
if(LENIA_BUILD_TESTS)
    add_executable(lenia_tests
        tests/core/test_grid.cc
        tests/core/test_kernel.cc
        tests/core/test_fft.cc
        tests/core/test_growth.cc
        tests/core/test_lenia.cc
    )
    target_link_libraries(lenia_tests PRIVATE lenia_core gtest gtest_main)
    target_include_directories(lenia_tests PRIVATE src)

    include(GoogleTest)
    gtest_discover_tests(lenia_tests)
endif()
