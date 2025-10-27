#include "../include/utils.hpp" // Inclui as declarações
#include <vector>
#include <cmath>     // Para std::abs, std::sqrt
#include <iostream>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <omp.h>     // Para OpenMP

// --- Implementação da Função de Normalização ---

template<typename MatrixType>
void normalize_system_rows(MatrixType &H, Eigen::VectorXd &g) {
    if (H.rows() != g.size()) {
        throw std::runtime_error(
            "normalize_system_rows: Dimensoes H/g incompativeis. H.rows=" + std::to_string(H.rows()) + ", g.size=" +
            std::to_string(g.size()));
    }
    if (H.rows() == 0) {
        std::cout << "[INFO] Matriz H vazia, nada para normalizar." << std::endl;
        return; // Nada a fazer
    }

    std::cout << "[INFO] Normalizando linhas de H e elementos de g..." << std::endl;
    constexpr double epsilon_norm = 1e-12; // Limiar para evitar divisao por zero

    if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
        // --- Versão Densa ---
#pragma omp parallel for schedule(static) // Paraleliza a normalização das linhas
        for (Eigen::Index i = 0; i < H.rows(); ++i) {
            double row_norm = H.row(i).norm();
            if (row_norm > epsilon_norm) {
                H.row(i) /= row_norm;
                g(i) /= row_norm;
            } else {
                // Linha é (quase) zero, zera explicitamente para evitar NaNs
                H.row(i).setZero();
                g(i) = 0.0;
            }
        }
    } else {
        // --- Versão Esparsa ---
        std::vector<double> row_norms_sq(H.rows(), 0.0);

        // Calcula norma ao quadrado de cada linha (iterando pelos não-zeros)
        // Esta parte é inerentemente sequencial por coluna na estrutura CSC do Eigen
        for (int k = 0; k < H.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(H, k); it; ++it) {
                row_norms_sq[it.row()] += it.value() * it.value();
            }
        }

        // Modifica os valores da matriz esparsa in-place (paralelizável por coluna)
#pragma omp parallel for schedule(static)
        for (int k = 0; k < H.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(H, k); it; ++it) {
                double row_norm = std::sqrt(row_norms_sq[it.row()]);
                if (row_norm > epsilon_norm) {
                    // valueRef() permite modificar o valor
                    it.valueRef() /= row_norm;
                } else {
                    // Se a norma da linha é zero, o valor deve ser zero
                    it.valueRef() = 0.0;
                }
            }
        }
        // Remove explicitamente os zeros que podem ter sido criados (opcional, mas bom para limpeza)
        H.prune(0.0, std::numeric_limits<double>::epsilon()); // Remove valores muito pequenos

        // Normaliza g (paralelizável)
#pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < g.size(); ++i) {
            double row_norm = std::sqrt(row_norms_sq[i]);
            if (row_norm > epsilon_norm) {
                g(i) /= row_norm;
            } else {
                g(i) = 0.0;
            }
        }
    }
    std::cout << "[INFO] Normalizacao concluida." << std::endl;
}

// --- Instanciação explícita para os tipos que vamos usar ---
// Isso garante que o compilador gere o código para MatrixXd e SparseMatrix<double>
// Coloque isso no final do arquivo .cpp
template void normalize_system_rows<Eigen::MatrixXd>(Eigen::MatrixXd &H, Eigen::VectorXd &g);

template void normalize_system_rows<Eigen::SparseMatrix<double> >(Eigen::SparseMatrix<double> &H, Eigen::VectorXd &g);

// --- Implementação de Outras Funções Utilitárias (se houver) ---
// Exemplo: isToeplitz (copiado do solver_comparison.hpp anterior)
/*
template<typename MatrixType>
bool isToeplitz(const MatrixType &H, const double tolerance = 1e-10) {
    const Eigen::Index rows = H.rows();
    const Eigen::Index cols = H.cols();

    if (rows == 0 || cols == 0) return true;

    for (int d = -rows + 1; d < cols; ++d) {
        double first_val = 0.0; // Initialize properly
        bool first = true;

        for (Eigen::Index i = std::max(0, -d); i < std::min(rows, cols - d); ++i) {
            Eigen::Index j = i + d;
            double current_val;

            if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
                current_val = H(i, j);
            } else {
                current_val = H.coeff(i, j); // Use coeff for sparse
            }

            if (first) {
                 // Only record the first non-zero (or near-zero) value encountered on the diagonal
                 if (std::abs(current_val) > tolerance * 1e-3) { // Use a smaller tolerance to find the 'first' meaningful value
                     first_val = current_val;
                     first = false;
                 } else if (i == std::min(rows, cols - d) - 1) { // If we reach the end and only found zeros
                     first_val = 0.0; // Consider the diagonal constant zero
                     first = false; // Prevent comparison below
                 }
            } else {
                // Only compare if the current value is also significant
                if (std::abs(current_val) > tolerance * 1e-3 || std::abs(first_val) > tolerance * 1e-3) {
                     if (std::abs(current_val - first_val) > tolerance) {
                         //std::cout << "[DEBUG] Diagonal " << d << " nao e constante na pos (" << i << "," << j << "): "
                         //          << first_val << " vs " << current_val << std::endl;
                         return false;
                     }
                }
                 // If both current_val and first_val are near zero, consider them equal
            }
        }
    }
    return true;
}

// Instanciações explícitas para isToeplitz
template bool isToeplitz<Eigen::MatrixXd>(const Eigen::MatrixXd& H, double tolerance);
template bool isToeplitz<Eigen::SparseMatrix<double>>(const Eigen::SparseMatrix<double>& H, double tolerance);
*/
