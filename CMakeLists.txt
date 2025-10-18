# The MiniCppStarter for Ultrasound Reconstruction
#
# Build and run by calling:
# cmake -S. -B build && cmake --build build && ./build/Playground

cmake_minimum_required(VERSION 3.16)

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

# OpenMP
find_package(OpenMP REQUIRED)

# ---- Create executable ----

# O nome do executável agora é mais descritivo
add_executable(UltrasoundServer main.cpp)

# Linkamos com a biblioteca Eigen.
# O alvo 'Eigen3::Eigen' é criado pelo CMake do Eigen e inclui os diretórios de header.
target_link_libraries(UltrasoundServer PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

# Configuramos para usar o padrão C++26, como desejado.
set_target_properties(UltrasoundServer PROPERTIES CXX_STANDARD 26)