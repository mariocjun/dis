# The MiniCppStarter for Ultrasound Reconstruction
#
# Build and run by calling:
# cmake -S. -B build && cmake --build build && ./build/Playground

cmake_minimum_required(VERSION 3.23)
cmake_policy(VERSION 3.23)

# Garante compatibilidade com pacotes antigos que pedem < 3.5
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Ativar log de comandos de compilação cmake
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE BOOL "Suprimir avisos de desenvolvedor")

# FORCE RELEASE BUILD TYPE IF NOT SPECIFIED
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# ---- Project ----

project(UltrasoundServer LANGUAGES CXX)

# ---- Fetch CPM ----
file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.39.0/CPM.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
        EXPECTED_HASH SHA256=66639BCAC9DD2907B2918DE466783554C1334446B9874E90D38E3778D404C2EF
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# ---- Add dependencies via CPM ----

# Eigen3
CPMAddPackage(
        NAME Eigen3
        VERSION 3.4.0
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
        GIT_TAG "3.4.0"
        OPTIONS
        "BUILD_TESTING OFF"
        "EIGEN_BUILD_DOC OFF"
)

# yaml-cpp
CPMAddPackage(
        NAME yaml-cpp
        GIT_REPOSITORY "https://github.com/jbeder/yaml-cpp.git"
        GIT_TAG "0.8.0"
        OPTIONS
        "YAML_CPP_BUILD_TESTS OFF"
        "YAML_CPP_BUILD_TOOLS OFF"
)

# OpenMP
find_package(OpenMP REQUIRED)

# --- BLAS / MKL Support (Bonus) ---
# Tenta encontrar MKL (Intel Math Kernel Library) para igualar Python
find_package(MKL QUIET)

if(MKL_FOUND)
    message(STATUS "🚀 Intel MKL encontrado! Ativando aceleração máxima.")
    add_compile_definitions(EIGEN_USE_MKL_ALL)
    
    # Tenta linkar via algos comuns de Config/Module
    if(TARGET MKL::MKL)
        link_libraries(MKL::MKL)
    elseif(TARGET MKL::MKL_64)
         link_libraries(MKL::MKL_64)
    else()
        # Fallback para variaveis se o target nao existir
        if(MKL_INCLUDE_DIR)
            include_directories(${MKL_INCLUDE_DIR})
        endif()
        link_libraries(${MKL_LIBRARIES})
    endif()
else()
    message(STATUS "⚠️ Intel MKL não encontrado. Usando Eigen padrão (pode ser mais lento que Python MKL).")
    # Tentativa secundária: OpenBLAS
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        message(STATUS "🚀 BLAS encontrado! Ativando aceleração BLAS.")
        add_compile_definitions(EIGEN_USE_BLAS)
        link_libraries(${BLAS_LIBRARIES})
    endif()
endif()
# ----------------------------------

# nlohmann_json
CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.11.2
)

# httplib REMOVIDO - usando sockets raw

# Python3
find_package(Python3 REQUIRED)

# ---- Create executable ----
add_executable(UltrasoundBenchmark
        main.cpp
        src/config.cpp
        src/io_utils.cpp
        src/solvers.cpp
        src/utils.cpp
        src/benchmark_runner.cpp
        src/reporting.cpp
)

# ---- Link Libraries ----
target_link_libraries(UltrasoundBenchmark PRIVATE
        Eigen3::Eigen       # Eigen
        yaml-cpp::yaml-cpp  # yaml-cpp
        OpenMP::OpenMP_CXX  # OpenMP
        nlohmann_json::nlohmann_json
)

# httplib removido - usando sockets raw

if(WIN32)
    target_link_libraries(UltrasoundBenchmark PRIVATE ws2_32)
endif()

# Necessário para std::filesystem no GCC < 9
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(UltrasoundBenchmark PRIVATE -lstdc++fs)
endif ()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    target_link_libraries(UltrasoundBenchmark PRIVATE -lc++fs)
endif ()


# ---- Post-Build Command ----
# add_custom_command(
#         TARGET UltrasoundBenchmark
#         POST_BUILD
#         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/documentar.py
#         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#         COMMENT "Gerando documentação do projeto via documentar.py..."
# )


# ---- Target Properties ----
set_target_properties(UltrasoundBenchmark PROPERTIES CXX_STANDARD 23)

# FORCE MAXIMUM OPTIMIZATION FOR ALL BUILDS
# Use aggressive optimizations unconditionally for PERFORMANCE DEMONSTRATION

# Generate build info string
string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S")
set(BUILD_INFO "Release+Optimized (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}) ${BUILD_TIMESTAMP}")

if(MSVC)
    # Apply optimizations only if NOT in Debug mode
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
    if(NOT "${_build_type_upper}" STREQUAL "DEBUG")
    # MSVC optimizations - MAXIMUM PERFORMANCE
    target_compile_options(UltrasoundBenchmark PRIVATE
        /O2           # Maximum optimization
        /Ob3          # Aggressive inlining
        /Oi           # Enable intrinsics
        /Ot           # Favor fast code
        /Oy           # Omit frame pointers
        /fp:fast      # Fast floating point
        /GL           # Whole program optimization
        /GS-          # Disable security checks (performance)
        /arch:AVX2    # Use AVX2 SIMD instructions
        /DNDEBUG      # Disable asserts
        /openmp       # OpenMP support
    )
    target_link_options(UltrasoundBenchmark PRIVATE /LTCG /OPT:REF /OPT:ICF)
    target_compile_definitions(UltrasoundBenchmark PRIVATE 
        BUILD_INFO="${BUILD_INFO}"
        OPTIMIZED_BUILD=1
    )
    endif()
else()
    # GCC/Clang optimizations - MAXIMUM PERFORMANCE
    target_compile_options(UltrasoundBenchmark PRIVATE
        -O3                    # Maximum optimization
        -march=native          # Use native CPU features (SSE, AVX, etc)
        -mtune=native          # Tune for native CPU
        -ffast-math            # Fast floating point
        -funroll-loops         # Unroll loops
        -ftree-vectorize       # Enable vectorization
        -fomit-frame-pointer   # Omit frame pointers
        -flto                  # Link time optimization
        -DNDEBUG               # Disable asserts
        -fopenmp               # OpenMP support
    )
    target_link_options(UltrasoundBenchmark PRIVATE -flto)
    target_compile_definitions(UltrasoundBenchmark PRIVATE 
        BUILD_INFO="${BUILD_INFO}"
        OPTIMIZED_BUILD=1
    )
endif()

message(STATUS "=== PERFORMANCE BUILD ENABLED ===")
message(STATUS "Build Info: ${BUILD_INFO}")
message(STATUS "Optimizations: AVX2 SIMD, LTO, Fast Math, OpenMP")