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

# nlohmann_json
CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.11.2
)

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
add_custom_command(
        TARGET UltrasoundBenchmark
        POST_BUILD
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/documentar.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Gerando documentação do projeto via documentar.py..."
)


# ---- Target Properties ----
set_target_properties(UltrasoundBenchmark PROPERTIES CXX_STANDARD 23)

target_compile_options(UltrasoundBenchmark PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
target_compile_options(UltrasoundBenchmark PRIVATE $<$<CONFIG:Debug>:-g>)