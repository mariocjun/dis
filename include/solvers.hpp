#ifndef ULTRASOUNDBENCHMARK_SOLVERS_HPP
#define ULTRASOUNDBENCHMARK_SOLVERS_HPP

#include "types.hpp" // Para ReconstructionResult
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <string>
#include <filesystem> // Para std::filesystem::path

// --- Declarações dos Solvers ---

/**
 * @brief Executa o solver CGNR padrão com Regularização de Tikhonov (Salva Imagens Intermediárias).
 * Assume que H e g já foram normalizados. Para quando epsilon < tolerance ou max_iterations.
 * @param g_signal Sinal medido (vetor g) normalizado.
 * @param H_model Matriz do sistema (H) normalizada.
 * @param tolerance Tolerância para o critério de parada epsilon.
 * @param max_iterations Número máximo de iterações.
 * @param base_filename_prefix Prefixo para os nomes dos arquivos CSV das imagens intermediárias.
 * @param output_dir Diretório para salvar as imagens intermediárias.
 * @param img_rows Número de linhas da imagem para salvar CSV.
 * @param img_cols Número de colunas da imagem para salvar CSV.
 * @return ReconstructionResult Contendo a imagem final, métricas e histórico.
 */
template<typename MatrixType>
ReconstructionResult run_cgnr_solver_epsilon_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix,
    const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

/**
 * @brief Executa o solver CGNR com Pré-condicionador Jacobi e Regularização de Tikhonov (Salva Imagens Intermediárias).
 * Assume que H e g já foram normalizados. Para quando epsilon < tolerance ou max_iterations.
 * @param g_signal Sinal medido (vetor g) normalizado.
 * @param H_model Matriz do sistema (H) normalizada.
 * @param tolerance Tolerância para o critério de parada epsilon.
 * @param max_iterations Número máximo de iterações.
 * @param base_filename_prefix Prefixo para os nomes dos arquivos CSV das imagens intermediárias.
 * @param output_dir Diretório para salvar as imagens intermediárias.
 * @param img_rows Número de linhas da imagem para salvar CSV.
 * @param img_cols Número de colunas da imagem para salvar CSV.
 * @return ReconstructionResult Contendo a imagem final, métricas e histórico.
 */
template<typename MatrixType>
ReconstructionResult run_cgnr_solver_preconditioned_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix,
    const std::filesystem::path& output_dir,
    int img_rows, int img_cols);


/**
 * @brief Executa o solver CGNR padrão (sem regularização explícita aqui) por um número FIXO de iterações.
 * Usado para gerar dados consistentes para os CSVs de convergência e L-curve.
 * Assume que H e g já foram normalizados. NÃO salva imagens intermediárias.
 * @param g_signal Sinal medido (vetor g) normalizado.
 * @param H_model Matriz do sistema (H) normalizada.
 * @param num_iterations Número exato de iterações a executar.
 * @return ReconstructionResult Contendo histórico de resíduo e solução.
 */
template<typename MatrixType>
ReconstructionResult run_cgnr_solver_fixed_iter(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    int num_iterations);


/**
 * @brief Executa o solver FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) com salvamento de iterações.
 * Usa regularização L1 (soft thresholding) e atualização de momento FISTA.
 * @param g_signal Sinal medido (vetor g) normalizado.
 * @param H_model Matriz do sistema (H) normalizada.
 * @param tolerance Tolerância para o critério de parada epsilon.
 * @param max_iterations Número máximo de iterações.
 * @param base_filename_prefix Prefixo para os nomes dos arquivos CSV das imagens intermediárias.
 * @param output_dir Diretório para salvar as imagens intermediárias.
 * @param img_rows Número de linhas da imagem para salvar CSV.
 * @param img_cols Número de colunas da imagem para salvar CSV.
 * @return ReconstructionResult Contendo a imagem final, métricas e histórico.
 */
template<typename MatrixType>
ReconstructionResult run_fista_solver_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix,
    const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

#endif //ULTRASOUNDBENCHMARK_SOLVERS_HPP