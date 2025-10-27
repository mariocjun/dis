#ifndef ULTRASOUNDBENCHMARK_UTILS_HPP
#define ULTRASOUNDBENCHMARK_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <string>

// --- Declaração da Função de Normalização ---

/**
 * @brief Normaliza as linhas da matriz H e os elementos correspondentes do vetor g.
 * Modifica H e g no local.
 * @tparam MatrixType Eigen::MatrixXd ou Eigen::SparseMatrix<double>
 * @param H Matriz do sistema (será modificada).
 * @param g Vetor do sinal medido (será modificado).
 */
template<typename MatrixType>
void normalize_system_rows(MatrixType& H, Eigen::VectorXd& g);

// --- Declaração de Outras Funções Utilitárias (se houver) ---
// Exemplo: Função para verificar estrutura Toeplitz (se ainda for útil para análise)
// template<typename MatrixType>
// bool isToeplitz(const MatrixType& H, double tolerance = 1e-10);


#endif //ULTRASOUNDBENCHMARK_UTILS_HPP