# The MiniCppStarter for Ultrasound Reconstruction
#
# Build and run by calling:
# cmake -S. -B build && cmake --build build && ./build/Playground

cmake_minimum_required(VERSION 3.23)
cmake_policy(VERSION 3.23)

include_guard(GLOBAL)

# Garante compatibilidade com pacotes antigos que pedem < 3.5
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Ativar log de comandos de compilação cmake
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE BOOL "Suprimir avisos de desenvolvedor")

# ---- Project ----

project(UltrasoundServer LANGUAGES CXX)

# ---- Fetch CPM ----
# (Esta parte baixa o gerenciador de pacotes CPM, mantemos como está)
file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.39.0/CPM.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
        EXPECTED_HASH SHA256=66639BCAC9DD2907B2918DE466783554C1334446B9874E90D38E3778D404C2EF
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# ---- Add dependencies via CPM ----

# MUDANÇA: Usando o repositório oficial do GitLab e a versão 5.0.0.
# O CPM fará o download do código-fonte. Como Eigen é "header-only",
# não há etapa de compilação para a própria biblioteca.
CPMAddPackage(
        NAME Eigen3
        VERSION 5.0.0
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
        GIT_TAG "5.0.0"
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

# cpp-httplib (header-only HTTP server library)
CPMAddPackage(
        NAME httplib
        GIT_REPOSITORY "https://github.com/yhirose/cpp-httplib.git"
        GIT_TAG "v0.18.3"
)

# nlohmann/json (header-only JSON library)
CPMAddPackage(
        NAME nlohmann_json
        GIT_REPOSITORY "https://github.com/nlohmann/json.git"
        GIT_TAG "v3.11.3"
)

# --- NOVO BLOCO 1: Encontrar o Python ---
# Adicionado para podermos chamar o script de documentação
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
)

# Necessário para std::filesystem no GCC < 9
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(UltrasoundBenchmark PRIVATE -lstdc++fs)
endif ()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    target_link_libraries(UltrasoundBenchmark PRIVATE -lc++fs)
endif ()


# --- NOVO BLOCO 2: Comando Pós-Compilação ---
# Executa o script documentar.py toda vez que o 'UltrasoundBenchmark'
# for compilado com sucesso.
add_custom_command(
        TARGET UltrasoundBenchmark
        POST_BUILD  # Executa APÓS a compilação
        COMMAND
        # O comando que será executado no terminal
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/documentar.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Gerando documentação do projeto via documentar.py..."
)


# ---- Target Properties ----
set_target_properties(UltrasoundBenchmark PROPERTIES CXX_STANDARD 23)

# Compiler-specific optimization flags
if(MSVC)
    target_compile_options(UltrasoundBenchmark PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(UltrasoundBenchmark PRIVATE $<$<CONFIG:Debug>:/Zi>)
else()
    target_compile_options(UltrasoundBenchmark PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
    target_compile_options(UltrasoundBenchmark PRIVATE $<$<CONFIG:Debug>:-g>)
endif()

# ---- HTTP Server Executable ----
add_executable(UltrasoundServerHTTP
        server/cpp_http_server.cpp
        src/config.cpp
        src/io_utils.cpp
        src/solvers.cpp
        src/utils.cpp
)

# Include directories for HTTP server
target_include_directories(UltrasoundServerHTTP PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${httplib_SOURCE_DIR}
        ${nlohmann_json_SOURCE_DIR}/include
)

# Link libraries for HTTP server
target_link_libraries(UltrasoundServerHTTP PRIVATE
        Eigen3::Eigen
        yaml-cpp::yaml-cpp
        OpenMP::OpenMP_CXX
)

# Windows needs ws2_32 for networking
if(WIN32)
    target_link_libraries(UltrasoundServerHTTP PRIVATE ws2_32)
endif()

# Filesystem for older compilers
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(UltrasoundServerHTTP PRIVATE -lstdc++fs)
endif ()

# Set C++ standard
set_target_properties(UltrasoundServerHTTP PROPERTIES CXX_STANDARD 23)

# Compiler-specific optimization flags
if(MSVC)
    target_compile_options(UltrasoundServerHTTP PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(UltrasoundServerHTTP PRIVATE $<$<CONFIG:Debug>:/Zi>)
else()
    target_compile_options(UltrasoundServerHTTP PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
    target_compile_options(UltrasoundServerHTTP PRIVATE $<$<CONFIG:Debug>:-g>)
endif()