#ifndef ULTRASOUNDSERVER_SOLVER_COMPARISON_HPP
#define ULTRASOUNDSERVER_SOLVER_COMPARISON_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


// Inclui as definições de structs do seu próprio projeto
#include "include/io_utils.hpp" // Incluir o cabeçalho com as declarações
#include "include/types.hpp"


// --- Funções Auxiliares (Implementações Completas) ---
// (As 11 funções: loadVectorData, convertCsvToSparse, saveSparseMatrix,
// loadSparseMatrix, loadDenseData, loadDenseMatrix, saveDenseMatrix,
// saveHistoryToCSV, saveLcurveToCSV, saveImageVectorToCsv,
// normalize_system_rows... ESTÃO AQUI)

template <typename MatrixType>
inline void normalize_system_rows(MatrixType &H, Eigen::VectorXd &g) {
  if (H.rows() != g.size()) {
    throw std::runtime_error(
        "normalize_system_rows: Dimensoes H/g incompativeis.");
  }
  if (H.rows() == 0)
    return;

  std::cout << "[INFO] Normalizando linhas de H e elementos de g..."
            << std::endl;
  constexpr double epsilon_norm = 1e-12;

  if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
#pragma omp parallel for
    for (Eigen::Index i = 0; i < H.rows(); ++i) {
      double row_norm = H.row(i).norm();
      if (row_norm > epsilon_norm) {
        H.row(i) /= row_norm;
        g(i) /= row_norm;
      } else {
        H.row(i).setZero();
        g(i) = 0.0;
      }
    }
  } else {
    // Esparsa
    std::vector<double> row_norms_sq(H.rows(), 0.0);
    for (int k = 0; k < H.outerSize(); ++k) {
      for (typename MatrixType::InnerIterator it(H, k); it; ++it) {
        row_norms_sq[it.row()] += it.value() * it.value();
      }
    }
    std::vector<Eigen::Triplet<double>> triplets_normalized;
    triplets_normalized.reserve(H.nonZeros());
    for (int k = 0; k < H.outerSize(); ++k) {
      for (typename MatrixType::InnerIterator it(H, k); it; ++it) {
        double row_norm = std::sqrt(row_norms_sq[it.row()]);
        if (row_norm > epsilon_norm) {
          triplets_normalized.emplace_back(it.row(), it.col(),
                                           it.value() / row_norm);
        }
      }
    }
    H.setFromTriplets(triplets_normalized.begin(), triplets_normalized.end());
#pragma omp parallel for
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

// --- Solver CGNR Regularizado (Salva Imagens Intermediárias) ---
template <typename MatrixType>
inline ReconstructionResult run_cgnr_solver_epsilon_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string &base_filename_prefix, const std::string &output_dir_str,
    int img_rows, int img_cols, bool save_intermediate_images) {
  if (H_model.rows() != g_signal.size())
    throw std::runtime_error(
        "Dimensoes incompativeis: H.rows()!=" + std::to_string(H_model.rows()) +
        " vs g.size()=" + std::to_string(g_signal.size()));
  if (H_model.cols() <= 0)
    throw std::runtime_error("Matriz H tem " + std::to_string(H_model.cols()) +
                             " colunas.");
  if (H_model.rows() == 0)
    return ReconstructionResult{};

  const auto start_time = std::chrono::high_resolution_clock::now();
  const std::filesystem::path output_dir(output_dir_str);

  Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
  Eigen::VectorXd r = g_signal;
  Eigen::VectorXd z = H_model.transpose() * r;
  Eigen::VectorXd p = z;
  double z_norm_sq = z.squaredNorm();

  double lambda = 0.0;
  if (z.size() > 0) {
    lambda = z.cwiseAbs().maxCoeff() * 0.10;
    constexpr double min_lambda = 1e-9;
    if (lambda < min_lambda) {
      lambda = min_lambda;
      std::cout
          << "[INFO] Lambda calculado era quase zero, usando piso minimo: "
          << lambda << std::endl;
    }
  } else {
    lambda = 1e-9;
    std::cout << "[AVISO] Vetor z inicial vazio, usando lambda=" << lambda
              << " como fallback." << std::endl;
  }
  std::cout << "[INFO] Lambda (solver standard): " << lambda << std::endl;

  ReconstructionResult result;
  result.iterations = 0;
  result.converged = false;
  result.residual_history.clear();
  result.residual_history.reserve(max_iterations);
  result.solution_history.clear();
  result.solution_history.reserve(max_iterations);
  double previous_residual_norm = r.norm();
  double current_residual_norm = previous_residual_norm;
  double epsilon = std::numeric_limits<double>::max();

  bool save_iters = save_intermediate_images && !base_filename_prefix.empty() &&
                    img_rows > 0 && img_cols > 0;
  if (save_iters) {
    try {
      std::filesystem::path iter_img_path =
          output_dir / "images" / (base_filename_prefix + "_iter_0.csv");
      saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
    } catch (const std::exception &e) {
      std::cerr << "[AVISO] Falha ao salvar imagem iter 0: " << e.what()
                << std::endl;
    }
  }

  for (int i = 0; i < max_iterations; ++i) {
    result.iterations = i + 1;
    Eigen::VectorXd w = H_model * p;
    double p_norm_sq = p.squaredNorm();
    double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

    if (modified_denominator < std::numeric_limits<double>::epsilon()) {
      std::cout
          << "[INFO] Denominador modificado (...) proximo de zero na iteracao "
          << i + 1 << ". Parando." << std::endl;
      break;
    }

    double alpha = z_norm_sq / modified_denominator;
    f += alpha * p;
    r -= alpha * w;

    if (save_iters) {
      try {
        std::filesystem::path iter_img_path =
            output_dir / "images" /
            (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
        saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
      } catch (const std::exception &e) {
        std::cerr << "[AVISO] Falha ao salvar imagem iter " << i + 1 << ": "
                  << e.what() << std::endl;
      }
    }

    Eigen::VectorXd z_next = (H_model.transpose() * r) - (lambda * f);
    const double z_next_norm_sq = z_next.squaredNorm();

    current_residual_norm = r.norm();
    result.residual_history.push_back(current_residual_norm);
    result.solution_history.push_back(f.norm());
    epsilon = std::abs(current_residual_norm - previous_residual_norm);

    if (epsilon < tolerance) {
      result.converged = true;
      std::cout << "[INFO] Convergencia por epsilon atingida na iteracao "
                << i + 1 << " (epsilon=" << std::scientific << epsilon << " < "
                << tolerance << ")" << std::defaultfloat << std::endl;
      break;
    }
    previous_residual_norm = current_residual_norm;

    double beta = 0.0;
    if (z_norm_sq >= std::numeric_limits<double>::epsilon()) {
      beta = z_next_norm_sq / z_norm_sq;
    } else {
      std::cout << "[INFO] ||z||^2 (...) proximo de zero na iteracao " << i + 1
                << ". Usando beta=0 (restart)." << std::endl;
      if (z_next_norm_sq < std::numeric_limits<double>::epsilon()) {
        std::cout << "[INFO] ||z_next||^2 tambem proximo de zero. "
                     "Provavelmente estagnou. Parando."
                  << std::endl;
        break;
      }
    }

    p = z_next + beta * p;
    z = z_next;
    z_norm_sq = z_next_norm_sq;

    if (i == max_iterations - 1 && !result.converged) {
      std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations
                << ") atingido sem convergencia por epsilon (ultimo epsilon="
                << std::scientific << epsilon << std::defaultfloat << ")."
                << std::endl;
    }
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  result.image = f;
  result.final_error = current_residual_norm;
  result.final_epsilon = epsilon;
  result.execution_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  return result;
}

// --- Solver CGNR Pré-condicionado (Salva Imagens Intermediárias) ---
template <typename MatrixType>
inline ReconstructionResult run_cgnr_solver_preconditioned_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string &base_filename_prefix, const std::string &output_dir_str,
    int img_rows, int img_cols, bool save_intermediate_images) {
  if (H_model.rows() != g_signal.size())
    throw std::runtime_error("...");
  if (H_model.cols() <= 0)
    throw std::runtime_error("...");
  if (H_model.rows() == 0)
    return ReconstructionResult{};

  const auto start_time = std::chrono::high_resolution_clock::now();
  const std::filesystem::path output_dir(output_dir_str);

  Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
  Eigen::VectorXd r = g_signal;
  Eigen::VectorXd z_unprec = H_model.transpose() * r;

  Eigen::VectorXd preconditioner = Eigen::VectorXd::Ones(H_model.cols());
  if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
#pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < H_model.cols(); ++j) {
      preconditioner(j) = H_model.col(j).squaredNorm();
    }
  } else {
    // Esparsa
    preconditioner.setZero();
    for (int k = 0; k < H_model.outerSize(); ++k) {
      for (typename MatrixType::InnerIterator it(H_model, k); it; ++it) {
        preconditioner(it.col()) += it.value() * it.value();
      }
    }
  }
  preconditioner = preconditioner.cwiseMax(1e-12).cwiseInverse();
  std::cout << "[INFO] Pre-condicionador Jacobi calculado." << std::endl;

  Eigen::VectorXd z = z_unprec.cwiseProduct(preconditioner);
  Eigen::VectorXd p = z;
  double z_precond_dot_z = z_unprec.dot(z);

  double lambda = 0.0;
  if (z_unprec.size() > 0) {
    lambda = z_unprec.cwiseAbs().maxCoeff() * 0.10;
    constexpr double min_lambda = 1e-9;
    if (lambda < min_lambda) {
      lambda = min_lambda;
      std::cout
          << "[INFO] Lambda calculado era quase zero, usando piso minimo: "
          << lambda << std::endl;
    }
  } else {
    lambda = 1e-9;
    std::cout << "[AVISO] Vetor z inicial (unprec) vazio, usando lambda="
              << lambda << " como fallback." << std::endl;
  }
  std::cout << "[INFO] Lambda (solver precond): " << lambda << std::endl;

  ReconstructionResult result;
  result.iterations = 0;
  result.converged = false;
  result.residual_history.clear();
  result.residual_history.reserve(max_iterations);
  result.solution_history.clear();
  result.solution_history.reserve(max_iterations);
  double previous_residual_norm = r.norm();
  double current_residual_norm = previous_residual_norm;
  double epsilon = std::numeric_limits<double>::max();

  bool save_iters = save_intermediate_images && !base_filename_prefix.empty() &&
                    img_rows > 0 && img_cols > 0;
  if (save_iters) {
    try {
      std::filesystem::path iter_img_path =
          output_dir / "images" / (base_filename_prefix + "_iter_0.csv");
      saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
    } catch (const std::exception &e) {
      std::cerr << "[AVISO] Falha ao salvar imagem iter 0 (precond): "
                << e.what() << std::endl;
    }
  }

  for (int i = 0; i < max_iterations; ++i) {
    result.iterations = i + 1;
    Eigen::VectorXd w = H_model * p;
    double p_norm_sq = p.squaredNorm();
    double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

    if (modified_denominator < std::numeric_limits<double>::epsilon()) {
      std::cout
          << "[INFO] Denominador modificado (...) proximo de zero na iteracao "
          << i + 1 << ". Parando." << std::endl;
      break;
    }

    double alpha = z_precond_dot_z / modified_denominator;
    f += alpha * p;
    r -= alpha * w;

    if (save_iters) {
      try {
        std::filesystem::path iter_img_path =
            output_dir / "images" /
            (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
        saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
      } catch (const std::exception &e) {
        std::cerr << "[AVISO] Falha ao salvar imagem iter " << i + 1
                  << " (precond): " << e.what() << std::endl;
      }
    }

    Eigen::VectorXd z_next_unprec = (H_model.transpose() * r) - (lambda * f);
    Eigen::VectorXd z_next = z_next_unprec.cwiseProduct(preconditioner);
    double z_next_precond_dot_z_next = z_next_unprec.dot(z_next);

    current_residual_norm = r.norm();
    result.residual_history.push_back(current_residual_norm);
    result.solution_history.push_back(f.norm());
    epsilon = std::abs(current_residual_norm - previous_residual_norm);

    if (epsilon < tolerance) {
      result.converged = true;
      std::cout << "[INFO] Convergencia por epsilon atingida na iteracao "
                << i + 1 << " (epsilon=" << std::scientific << epsilon << " < "
                << tolerance << ")" << std::defaultfloat << std::endl;
      break;
    }
    previous_residual_norm = current_residual_norm;

    double beta = 0.0;
    if (z_precond_dot_z >= std::numeric_limits<double>::epsilon()) {
      beta = z_next_precond_dot_z_next / z_precond_dot_z;
    } else {
      std::cout << "[INFO] ||z||^2 (...) proximo de zero na iteracao " << i + 1
                << ". Usando beta=0 (restart)." << std::endl;
      if (z_next_precond_dot_z_next < std::numeric_limits<double>::epsilon()) {
        std::cout << "[INFO] ||z_next||^2 tambem proximo de zero. "
                     "Provavelmente estagnou. Parando."
                  << std::endl;
        break;
      }
    }

    p = z_next + beta * p;
    z = z_next;
    z_precond_dot_z = z_next_precond_dot_z_next;

    if (i == max_iterations - 1 && !result.converged) {
      std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations
                << ") atingido sem convergencia por epsilon (ultimo epsilon="
                << std::scientific << epsilon << std::defaultfloat << ")."
                << std::endl;
    }
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  result.image = f;
  result.final_error = current_residual_norm;
  result.final_epsilon = epsilon;
  result.execution_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  return result;
}

// Solver Fixo (NÃO salva imagens intermediárias)
template <typename MatrixType>
inline ReconstructionResult
run_cgnr_solver_fixed_iter(const Eigen::VectorXd &g_signal,
                           const MatrixType &H_model,
                           const int num_iterations) {
  if (H_model.rows() != g_signal.size())
    throw std::runtime_error("...");
  if (H_model.cols() <= 0)
    throw std::runtime_error("...");
  if (H_model.rows() == 0)
    return ReconstructionResult{};

  const auto start_time = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
  Eigen::VectorXd r = g_signal;
  Eigen::VectorXd z = H_model.transpose() * r;
  Eigen::VectorXd p = z;
  double z_norm_sq = z.squaredNorm();
  ReconstructionResult result;
  result.iterations = 0;
  result.converged = false;
  result.residual_history.clear();
  result.residual_history.reserve(num_iterations);
  result.solution_history.clear();
  result.solution_history.reserve(num_iterations);

  for (int i = 0; i < num_iterations; ++i) {
    result.iterations = i + 1;
    double current_residual_norm = r.norm();
    result.residual_history.push_back(current_residual_norm);
    double current_solution_norm = f.norm();
    result.solution_history.push_back(current_solution_norm);
    Eigen::VectorXd w = H_model * p;
    double w_norm_sq = w.squaredNorm();
    double alpha = 0.0;
    if (w_norm_sq >= std::numeric_limits<double>::epsilon()) {
      alpha = z_norm_sq / w_norm_sq;
    } else {
      /* std::cout << "[AVISO - FixedIter] ||H*p||^2 proximo de zero..." <<
       * std::endl; */
      z_norm_sq = 0.0;
    }
    f += alpha * p;
    r -= alpha * w;
    Eigen::VectorXd z_next = H_model.transpose() * r;
    const double z_next_norm_sq = z_next.squaredNorm();
    double beta = 0.0;
    if (z_norm_sq >= std::numeric_limits<double>::epsilon()) {
      beta = z_next_norm_sq / z_norm_sq;
    } else {
      /* std::cout << "[AVISO - FixedIter] ||z||^2 proximo de zero..." <<
       * std::endl; */
    }
    p = z_next + beta * p;
    z = z_next;
    z_norm_sq = z_next_norm_sq;
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  result.image = f;
  result.final_error = r.norm();
  result.execution_time_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  return result;
}

// --- Instanciações explícitas para os tipos de matrizes ---
template ReconstructionResult
run_cgnr_solver_epsilon_save_iters<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::string &output_dir,
    int img_rows, int img_cols, bool save_intermediate_images);

template ReconstructionResult
run_cgnr_solver_preconditioned_save_iters<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::string &output_dir,
    int img_rows, int img_cols, bool save_intermediate_images);

template ReconstructionResult
run_cgnr_solver_fixed_iter<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    int num_iterations);

// Instanciações para Matriz Densa (para reativar testes densos se necessário)
template ReconstructionResult
run_cgnr_solver_epsilon_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::string &output_dir,
    int img_rows, int img_cols, bool save_intermediate_images);

template ReconstructionResult
run_cgnr_solver_preconditioned_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::string &output_dir,
    int img_rows, int img_cols, bool save_intermediate_images);

template ReconstructionResult
run_cgnr_solver_fixed_iter<Eigen::MatrixXd>(const Eigen::VectorXd &g_signal,
                                            const Eigen::MatrixXd &H_model,
                                            int num_iterations);

// --- Função Principal de Comparação Esparsa ---
// **** CORREÇÃO: TestConfig -> DatasetConfig ****
// Função para encontrar o ponto ótimo na curva L (declarada antes de ser usada)
inline int find_l_curve_corner(const std::vector<double> &residual_norms,
                               const std::vector<double> &solution_norms) {
  if (residual_norms.size() < 3)
    return residual_norms.size() - 1;

  // Usa log dos valores para melhor análise
  std::vector<double> log_residual(residual_norms.size());
  std::vector<double> log_solution(solution_norms.size());

  // Calcula os logs e normaliza
  const double min_res = *std::ranges::min_element(residual_norms);
  const double max_res = *std::ranges::max_element(residual_norms);
  const double min_sol = *std::ranges::min_element(solution_norms);
  const double max_sol = *std::ranges::max_element(solution_norms);

  for (size_t i = 0; i < residual_norms.size(); ++i) {
    log_residual[i] = (std::log(residual_norms[i]) - std::log(min_res)) /
                      (std::log(max_res) - std::log(min_res));
    log_solution[i] = (std::log(solution_norms[i]) - std::log(min_sol)) /
                      (std::log(max_sol) - std::log(min_sol));
  }

  // Encontra o ponto de máxima curvatura
  double max_curvature = -1;
  int corner_idx = 1;

  for (size_t i = 1; i < log_residual.size() - 1; ++i) {
    // Vetores para os pontos adjacentes
    double dx1 = log_residual[i] - log_residual[i - 1];
    double dy1 = log_solution[i] - log_solution[i - 1];
    double dx2 = log_residual[i + 1] - log_residual[i];
    double dy2 = log_solution[i + 1] - log_solution[i];

    // Normaliza os vetores
    double len1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
    double len2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

    if (len1 > 1e-10 && len2 > 1e-10) {
      dx1 /= len1;
      dy1 /= len1;
      dx2 /= len2;
      dy2 /= len2;

      // Calcula o ângulo entre os vetores
      double cos_theta = dx1 * dx2 + dy1 * dy2;
      double curvature = 1.0 - cos_theta; // Simplificação da curvatura

      if (curvature > max_curvature) {
        max_curvature = curvature;
        corner_idx = i;
      }
    }
  }

  return corner_idx;
}

inline std::pair<PerformanceMetrics, PerformanceMetrics>
run_sparse_comparison(const DatasetConfig &config,
                      const GlobalSettings &settings) {
  PerformanceMetrics standard_metrics;
  PerformanceMetrics precond_metrics;
  standard_metrics.optimization_type = "standard";
  precond_metrics.optimization_type = "jacobi";

  // Primeiro roda com iterações suficientes para encontrar o ponto ótimo
  constexpr int initial_iterations =
      50; // Número de iterações para construir a curva L
  const int max_iterations = settings.max_iterations;
  const double epsilon_tolerance = settings.epsilon_tolerance;

  std::cout << "\n-----------------------------------------------------"
            << std::endl;
  // **** CORREÇÃO: test_name -> description ****
  std::cout << "Iniciando Comparacao Esparsa para: " << config.description
            << std::endl;
  std::cout << "-----------------------------------------------------"
            << std::endl;

  // **** CORREÇÃO: test_name -> name ****
  std::string base_filename = config.name; // Usa o nome curto para arquivos
  // C++17 compatível replace:
  std::ranges::replace(base_filename, ' ', '_');
  std::ranges::replace(base_filename, '(', '_');
  std::ranges::replace(base_filename, ')', '_');
  std::ranges::replace(base_filename, '-', '_');

  std::filesystem::path output_dir = settings.output_base_dir;
  // **** CORREÇÃO: h_matrix_path -> h_matrix_csv ****
  std::filesystem::path h_path = config.h_matrix_csv;
  std::filesystem::path data_dir = h_path.parent_path();
  std::filesystem::path sparse_bin_fs_path =
      data_dir / (h_path.filename().string() + ".sparse.bin");

  // --- Teste 1: CGNR Padrão ---
  try {
    std::cout << "\n--- Rodando CGNR Esparso Padrao (Binario) ---\n";
    auto start_load = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> H_std =
        loadSparseMatrix(sparse_bin_fs_path.string());
    // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
    Eigen::VectorXd g_std = loadVectorData(config.g_signal_csv);
    auto end_load = std::chrono::high_resolution_clock::now();
    standard_metrics.load_time_ms =
        std::chrono::duration<double, std::milli>(end_load - start_load)
            .count();
    standard_metrics.estimated_ram_mb =
        static_cast<double>(H_std.nonZeros() * (sizeof(double) + sizeof(int)) +
                            (H_std.outerSize() + 1) * sizeof(int)) /
        (1024.0 * 1024.0);

    normalize_system_rows(H_std, g_std);
    Eigen::VectorXd z0_std = H_std.transpose() * g_std;
    std::cout << "[DEBUG Standard] Norma de z0 (H^T * g norm): "
              << z0_std.norm() << std::endl;

    // Define o prefixo do arquivo para o FISTA
    std::string filename_prefix_std =
        "image_" + base_filename + "_sparse_standard";
    std::string filename_prefix_fista =
        "image_" + base_filename + "_sparse_fista";
    ReconstructionResult res_std = run_cgnr_solver_epsilon_save_iters(
        g_std, H_std, epsilon_tolerance, max_iterations, filename_prefix_std,
        output_dir.string(), config.image_rows, config.image_cols,
        settings.save_intermediate_images);

    standard_metrics.solve_time_ms = res_std.execution_time_ms;
    standard_metrics.iterations = res_std.iterations;
    standard_metrics.final_error = res_std.final_error;
    standard_metrics.final_epsilon = res_std.final_epsilon;
    standard_metrics.converged = res_std.converged;

    // Primeiro executa com iterações suficientes para construir a curva L
    ReconstructionResult res_std_initial =
        run_cgnr_solver_fixed_iter(g_std, H_std, initial_iterations);

    // Encontra o ponto ótimo na curva L
    int optimal_iter = find_l_curve_corner(res_std_initial.residual_history,
                                           res_std_initial.solution_history);
    std::filesystem::path hist_path_std =
        output_dir /
        ("convergence_history_" + base_filename + "_sparse_standard.csv");
    std::filesystem::path lcurve_path_std =
        output_dir / ("lcurve_" + base_filename + "_sparse_standard.csv");
    saveHistoryToCSV(res_std_initial, hist_path_std.string());
    saveLcurveToCSV(res_std_initial, lcurve_path_std.string());

    // Executa novamente com o número ótimo de iterações
    std::cout << "[INFO] Ponto otimo da curva L encontrado na iteracao "
              << optimal_iter << std::endl;
    ReconstructionResult res_std_optimal =
        run_cgnr_solver_fixed_iter(g_std, H_std, optimal_iter);
  } catch (const std::exception &e) {
    std::cerr << "[ERRO - Esparso Padrao] " << e.what() << std::endl;
    standard_metrics = PerformanceMetrics();
    standard_metrics.optimization_type = "standard";
  }

  // --- Teste 2: CGNR Pré-condicionado ---
  try {
    std::cout << "\n--- Rodando CGNR Esparso Pre-condicionado (Binario) ---\n";
    auto start_load = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> H_pre =
        loadSparseMatrix(sparse_bin_fs_path.string());
    // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
    Eigen::VectorXd g_pre = loadVectorData(config.g_signal_csv);
    auto end_load = std::chrono::high_resolution_clock::now();
    precond_metrics.load_time_ms =
        std::chrono::duration<double, std::milli>(end_load - start_load)
            .count();
    precond_metrics.estimated_ram_mb =
        static_cast<double>(H_pre.nonZeros() * (sizeof(double) + sizeof(int)) +
                            (H_pre.outerSize() + 1) * sizeof(int)) /
        (1024.0 * 1024.0);

    normalize_system_rows(H_pre, g_pre);
    Eigen::VectorXd z0_pre = H_pre.transpose() * g_pre;
    std::cout << "[DEBUG Precond] Norma de z0 (H^T * g norm): " << z0_pre.norm()
              << std::endl;

    std::string filename_prefix_pre =
        "image_" + base_filename + "_sparse_precond";
    ReconstructionResult res_pre = run_cgnr_solver_preconditioned_save_iters(
        g_pre, H_pre, epsilon_tolerance, max_iterations, filename_prefix_pre,
        output_dir.string(), config.image_rows, config.image_cols,
        settings.save_intermediate_images);

    precond_metrics.solve_time_ms = res_pre.execution_time_ms;
    precond_metrics.iterations = res_pre.iterations;
    precond_metrics.final_error = res_pre.final_error;
    precond_metrics.final_epsilon = res_pre.final_epsilon;
    precond_metrics.converged = res_pre.converged;

    // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
    // Primeiro executa com iterações suficientes para construir a curva L
    ReconstructionResult res_pre_initial =
        run_cgnr_solver_fixed_iter(g_pre, H_pre, initial_iterations);

    // Encontra o ponto ótimo na curva L
    int optimal_iter_pre = find_l_curve_corner(
        res_pre_initial.residual_history, res_pre_initial.solution_history);
    std::cout << "[INFO] Ponto otimo da curva L (precondicionado) encontrado "
                 "na iteracao "
              << optimal_iter_pre << std::endl;

    // Executa novamente com o número ótimo de iterações
    ReconstructionResult res_pre_optimal =
        run_cgnr_solver_fixed_iter(g_pre, H_pre, optimal_iter_pre);
    std::filesystem::path hist_path_pre =
        output_dir /
        ("convergence_history_" + base_filename + "_sparse_precond.csv");
    // **** CORREÇÃO: Typo "lcurve_"to_string() ****
    std::filesystem::path lcurve_path_pre =
        output_dir / ("lcurve_" + base_filename + "_sparse_precond.csv");
    saveHistoryToCSV(res_pre_initial, hist_path_pre.string());
    saveLcurveToCSV(res_pre_initial, lcurve_path_pre.string());
  } catch (const std::exception &e) {
    std::cerr << "[ERRO - Esparso Precondicionado] " << e.what() << std::endl;
    precond_metrics = PerformanceMetrics();
    precond_metrics.optimization_type = "jacobi";
  }

  return {standard_metrics, precond_metrics};
}

// --- Criação de Diretórios de Saída ---
inline void create_output_directories(const std::filesystem::path &output_dir) {
  auto images_dir = output_dir / "images";
  auto metrics_dir = output_dir / "metrics";
  auto lcurve_dir = output_dir / "lcurve";

  std::filesystem::create_directories(images_dir);
  std::filesystem::create_directories(metrics_dir);
  std::filesystem::create_directories(lcurve_dir);

  std::cout << "[INFO] Diretorios de saida criados:" << std::endl;
  std::cout << "  - Imagens: " << images_dir << std::endl;
  std::cout << "  - Metricas: " << metrics_dir << std::endl;
  std::cout << "  - Curvas L: " << lcurve_dir << std::endl;
}

inline void save_iteration_image(const Eigen::VectorXd &vec,
                                 const std::filesystem::path &output_dir,
                                 const std::string &base_filename_prefix,
                                 int iteration, int img_rows, int img_cols) {
  auto images_dir = output_dir / "images";
  std::filesystem::create_directories(images_dir);
  std::filesystem::path iter_img_path =
      images_dir /
      (base_filename_prefix + "_iter_" + std::to_string(iteration) + ".csv");
  saveImageVectorToCsv(vec, iter_img_path.string(), img_rows, img_cols);
}

inline void save_convergence_data(const ReconstructionResult &result,
                                  const std::filesystem::path &output_dir,
                                  const std::string &base_filename) {
  auto metrics_dir = output_dir / "metrics";
  auto lcurve_dir = output_dir / "lcurve";

  std::filesystem::create_directories(metrics_dir);
  std::filesystem::create_directories(lcurve_dir);

  const std::filesystem::path hist_path =
      metrics_dir / ("convergence_history_" + base_filename + ".csv");
  const std::filesystem::path lcurve_path =
      lcurve_dir / ("lcurve_" + base_filename + ".csv");

  saveHistoryToCSV(result, hist_path.string());
  saveLcurveToCSV(result, lcurve_path.string());
}
#endif // ULTRASOUNDSERVER_SOLVER_COMPARISON_HPP
