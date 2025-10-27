#include "../include/solvers.hpp" // Inclui as declarações
#include "../include/io_utils.hpp" // Para saveImageVectorToCsv
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <fstream>

// --- Implementação do Solver CGNR Regularizado (Salva Imagens Intermediárias) ---
template<typename MatrixType>
ReconstructionResult run_cgnr_solver_epsilon_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string& base_filename_prefix,
    const std::filesystem::path& output_dir,
    int img_rows, int img_cols)
{
    // Verificações iniciais
    if (H_model.rows() != g_signal.size()) throw std::runtime_error("run_cgnr_solver_epsilon_save_iters: Dimensoes H/g incompativeis. H.rows=" + std::to_string(H_model.rows()) + ", g.size=" + std::to_string(g_signal.size()));
    if (H_model.cols() <= 0) throw std::runtime_error("run_cgnr_solver_epsilon_save_iters: Matriz H tem " + std::to_string(H_model.cols()) + " colunas.");
    if (H_model.rows() == 0) {
        std::cerr << "[AVISO] run_cgnr_solver_epsilon_save_iters: Matriz H ou sinal g estao vazios." << std::endl;
        return ReconstructionResult{};
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    // Inicialização
    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd r = g_signal;
    Eigen::VectorXd z = H_model.transpose() * r;
    Eigen::VectorXd p = z;
    double z_norm_sq = z.squaredNorm();

    // Cálculo do Lambda
    double lambda = 0.0;
    if (z.size() > 0) {
        lambda = z.cwiseAbs().maxCoeff() * 0.10;
        constexpr double min_lambda = 1e-9;
        if (lambda < min_lambda) { lambda = min_lambda; std::cout << "[INFO] Lambda calculado era quase zero, usando piso minimo: " << lambda << std::endl;}
    } else { lambda = 1e-9; std::cout << "[AVISO] Vetor z inicial vazio, usando lambda=" << lambda << " como fallback." << std::endl;}
    std::cout << "[INFO] Lambda (solver standard): " << lambda << std::endl;

    ReconstructionResult result;
    result.iterations = 0; result.converged = false;
    result.residual_history.clear(); result.residual_history.reserve(max_iterations);
    result.solution_history.clear(); result.solution_history.reserve(max_iterations);
    double previous_residual_norm = r.norm();
    double current_residual_norm = previous_residual_norm;
    double epsilon = std::numeric_limits<double>::max();

    // Salva imagem inicial (iter 0)
    bool save_iters = !base_filename_prefix.empty() && img_rows > 0 && img_cols > 0;
    if (save_iters) {
         try {
             std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_0.csv");
             saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
         } catch (const std::exception& e) {
              std::cerr << "[AVISO] Falha ao salvar imagem iter 0: " << e.what() << std::endl;
         }
    }

    // Loop CGNR
    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;
        Eigen::VectorXd w = H_model * p;
        double p_norm_sq = p.squaredNorm();
        double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

        if (modified_denominator < std::numeric_limits<double>::epsilon()) {
            std::cout << "[INFO] Denominador modificado (" << modified_denominator << ") proximo de zero na iteracao " << i + 1 << ". Parando." << std::endl;
            break;
        }

        double alpha = z_norm_sq / modified_denominator;
        f += alpha * p;
        r -= alpha * w;

        // Salva imagem intermediária
        if (save_iters) {
             try {
                std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
                saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
             } catch (const std::exception& e) {
                 std::cerr << "[AVISO] Falha ao salvar imagem iter " << i+1 << ": " << e.what() << std::endl;
             }
        }

        Eigen::VectorXd z_next = (H_model.transpose() * r) - (lambda * f);
        const double z_next_norm_sq = z_next.squaredNorm();

        current_residual_norm = r.norm(); result.residual_history.push_back(current_residual_norm);
        result.solution_history.push_back(f.norm());
        epsilon = std::abs(current_residual_norm - previous_residual_norm);

        if (epsilon < tolerance) {
            result.converged = true;
            std::cout << "[INFO] Convergencia por epsilon atingida na iteracao " << i+1 << " (epsilon=" << std::scientific << epsilon << " < " << tolerance << ")" << std::defaultfloat << std::endl;
            break;
        }
        previous_residual_norm = current_residual_norm;

        double beta = 0.0;
        if (z_norm_sq >= std::numeric_limits<double>::epsilon()) { beta = z_next_norm_sq / z_norm_sq; }
        else {
            std::cout << "[INFO] ||z||^2 (" << z_norm_sq << ") proximo de zero na iteracao " << i + 1 << ". Usando beta=0 (restart)." << std::endl;
             if (z_next_norm_sq < std::numeric_limits<double>::epsilon()) {
                 std::cout << "[INFO] ||z_next||^2 tambem proximo de zero. Provavelmente estagnou. Parando." << std::endl;
                 break;
             }
        }

        p = z_next + beta * p;
        z = z_next;
        z_norm_sq = z_next_norm_sq;

        if (i == max_iterations - 1 && !result.converged) {
            std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations << ") atingido sem convergencia por epsilon (ultimo epsilon=" << std::scientific << epsilon << std::defaultfloat << ")." << std::endl;
        }
    } // Fim do loop

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f; result.final_error = current_residual_norm; result.final_epsilon = epsilon;
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

// --- Implementação do Solver CGNR Pré-condicionado (Salva Imagens Intermediárias) ---
template<typename MatrixType>
inline ReconstructionResult run_cgnr_solver_preconditioned_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string& base_filename_prefix,
    const std::filesystem::path& output_dir,
    int img_rows, int img_cols)
{
    if (H_model.rows() != g_signal.size()) throw std::runtime_error("...");
    if (H_model.cols() <= 0) throw std::runtime_error("...");
    if (H_model.rows() == 0) return ReconstructionResult{};

    const auto start_time = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd r = g_signal;
    Eigen::VectorXd z_unprec = H_model.transpose() * r;

    // Cálculo do pré-condicionador Jacobi
    Eigen::VectorXd preconditioner = Eigen::VectorXd::Ones(H_model.cols());
     if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) { /* Densa */
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < H_model.cols(); ++j) {
            preconditioner(j) = H_model.col(j).squaredNorm();
        }
     } else { // Esparsa
         preconditioner.setZero();
         // Não dá para paralelizar facilmente por coluna em CSC, faz sequencial
         for (int k=0; k<H_model.outerSize(); ++k) {
             for (typename MatrixType::InnerIterator it(H_model,k); it; ++it) {
                 // it.col() é a coluna, it.value() é o valor
                 preconditioner(it.col()) += it.value() * it.value();
             }
         }
     }
    preconditioner = preconditioner.cwiseMax(1e-12).cwiseInverse(); // Inverte (e evita divisão por zero)
    std::cout << "[INFO] Pre-condicionador Jacobi calculado." << std::endl;

    Eigen::VectorXd z = z_unprec.cwiseProduct(preconditioner); // z_0 = M^-1 * (H^T * r_0)
    Eigen::VectorXd p = z; // p_0 = z_0
    double z_precond_dot_z = z_unprec.dot(z); // z_0^T * M^-1 * z_unprec_0

    double lambda = 0.0;
    if (z_unprec.size() > 0) { // Usa z_unprec (H^T g) para calcular lambda
        lambda = z_unprec.cwiseAbs().maxCoeff() * 0.10;
         constexpr double min_lambda = 1e-9;
        if (lambda < min_lambda) { lambda = min_lambda; std::cout << "[INFO] Lambda calculado era quase zero, usando piso minimo: " << lambda << std::endl;}
    } else { lambda = 1e-9; std::cout << "[AVISO] Vetor z inicial (unprec) vazio, usando lambda=" << lambda << " como fallback." << std::endl;}
    std::cout << "[INFO] Lambda (solver precond): " << lambda << std::endl;

    ReconstructionResult result;
    result.iterations = 0; result.converged = false;
    result.residual_history.clear(); result.residual_history.reserve(max_iterations);
    result.solution_history.clear(); result.solution_history.reserve(max_iterations);
    double previous_residual_norm = r.norm();
    double current_residual_norm = previous_residual_norm;
    double epsilon = std::numeric_limits<double>::max();

     // Salva imagem inicial (iter 0)
     bool save_iters = !base_filename_prefix.empty() && img_rows > 0 && img_cols > 0;
     if (save_iters) {
         try {
             std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_0.csv");
             saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
         } catch (const std::exception& e) {
             std::cerr << "[AVISO] Falha ao salvar imagem iter 0 (precond): " << e.what() << std::endl;
         }
     }

    // Loop CGNR Pré-condicionado
    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;
        Eigen::VectorXd w = H_model * p;
        double p_norm_sq = p.squaredNorm();
        double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

        if (modified_denominator < std::numeric_limits<double>::epsilon()) {
            std::cout << "[INFO] Denominador modificado (" << modified_denominator << ") proximo de zero na iteracao " << i + 1 << ". Parando." << std::endl;
            break;
        }

        double alpha = z_precond_dot_z / modified_denominator; // Usa produto interno modificado

        f += alpha * p;
        r -= alpha * w;

         // Salva imagem intermediária
         if (save_iters) {
             try {
                std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
                saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
             } catch (const std::exception& e) {
                 std::cerr << "[AVISO] Falha ao salvar imagem iter " << i+1 << " (precond): " << e.what() << std::endl;
             }
        }

        Eigen::VectorXd z_next_unprec = (H_model.transpose() * r) - (lambda * f);
        Eigen::VectorXd z_next = z_next_unprec.cwiseProduct(preconditioner); // Aplica pré-condicionador
        double z_next_precond_dot_z_next = z_next_unprec.dot(z_next); // Novo produto interno

        current_residual_norm = r.norm(); result.residual_history.push_back(current_residual_norm);
        result.solution_history.push_back(f.norm());
        epsilon = std::abs(current_residual_norm - previous_residual_norm);

        if (epsilon < tolerance) {
            result.converged = true;
            std::cout << "[INFO] Convergencia por epsilon atingida na iteracao " << i + 1 << " (epsilon=" << std::scientific << epsilon << " < " << tolerance << ")" << std::defaultfloat << std::endl;
            break;
        }
        previous_residual_norm = current_residual_norm;

        double beta = 0.0;
        // Beta usa os produtos internos modificados
        if (z_precond_dot_z >= std::numeric_limits<double>::epsilon()) {
             beta = z_next_precond_dot_z_next / z_precond_dot_z;
        } else {
             std::cout << "[INFO] Produto interno z^T M^-1 z_unprec (" << z_precond_dot_z << ") proximo de zero na iteracao " << i + 1 << ". Usando beta=0 (restart)." << std::endl;
            if (z_next_precond_dot_z_next < std::numeric_limits<double>::epsilon()) {
                std::cout << "[INFO] Produto interno z_next^T M^-1 z_unprec_next tambem proximo de zero. Provavelmente estagnou. Parando." << std::endl;
                break;
            }
        }

        p = z_next + beta * p; // p_{i+1} = z_{i+1} + beta_i * p_i
        z = z_next; // z é o pré-condicionado
        z_precond_dot_z = z_next_precond_dot_z_next; // Atualiza produto interno

        if (i == max_iterations - 1 && !result.converged) {
            std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations << ") atingido sem convergencia por epsilon (ultimo epsilon=" << std::scientific << epsilon << std::defaultfloat << ")." << std::endl;
        }
    } // Fim do loop

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f; result.final_error = current_residual_norm; result.final_epsilon = epsilon;
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}


// --- Implementação Solver Fixo (NÃO salva imagens intermediárias) ---
template<typename MatrixType>
inline ReconstructionResult run_cgnr_solver_fixed_iter(const Eigen::VectorXd &g_signal, const MatrixType &H_model,
                                                const int num_iterations) {
    if (H_model.rows() != g_signal.size()) throw std::runtime_error("run_cgnr_solver_fixed_iter: Dimensoes H/g incompativeis");
    if (H_model.cols() <= 0) throw std::runtime_error("run_cgnr_solver_fixed_iter: Matriz H tem 0 colunas");
    if (H_model.rows() == 0) return ReconstructionResult{};

    const auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd r = g_signal;
    Eigen::VectorXd z = H_model.transpose() * r;
    Eigen::VectorXd p = z;
    double z_norm_sq = z.squaredNorm();

    ReconstructionResult result;
    result.iterations = 0; result.converged = false;
    result.residual_history.clear(); result.residual_history.reserve(num_iterations);
    result.solution_history.clear(); result.solution_history.reserve(num_iterations);

    for (int i = 0; i < num_iterations; ++i) {
        result.iterations = i + 1;
        double current_residual_norm = r.norm(); result.residual_history.push_back(current_residual_norm);
        double current_solution_norm = f.norm(); result.solution_history.push_back(current_solution_norm);
        Eigen::VectorXd w = H_model * p; double w_norm_sq = w.squaredNorm();
        double alpha = 0.0;
        if (w_norm_sq >= std::numeric_limits<double>::epsilon()) { alpha = z_norm_sq / w_norm_sq; }
        else {
            // Não imprime aviso repetidamente, pode poluir muito
            // std::cout << "[AVISO - FixedIter] ||H*p||^2 proximo de zero na iteracao " << i + 1 << ". Usando alpha=0." << std::endl;
            z_norm_sq = 0.0;
        }
        f += alpha * p; r -= alpha * w;
        Eigen::VectorXd z_next = H_model.transpose() * r; const double z_next_norm_sq = z_next.squaredNorm();
        double beta = 0.0;
        if (z_norm_sq >= std::numeric_limits<double>::epsilon()) { beta = z_next_norm_sq / z_norm_sq; }
        else {
            // Não imprime aviso repetidamente
            // std::cout << "[AVISO - FixedIter] ||z||^2 proximo de zero na iteracao " << i + 1 << ". Usando beta=0." << std::endl;
        }
        p = z_next + beta * p; z = z_next; z_norm_sq = z_next_norm_sq;
    } // Fim do loop

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f; result.final_error = r.norm();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

template<typename MatrixType>
ReconstructionResult run_fista_solver_save_iters(const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    double tolerance, int max_iterations, const std::string &base_filename_prefix,
    const std::filesystem::path &output_dir, int img_rows, int img_cols) {

    const auto start_time = std::chrono::high_resolution_clock::now();
    ReconstructionResult result;
    result.converged = false;

    // Initialize variables
    const int n = H_model.cols();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);  // Current solution
    Eigen::VectorXd y = x;                         // Extrapolated point
    Eigen::VectorXd x_prev = x;                    // Previous solution
    double t = 1.0;                                // Initial step size
    double t_prev;
    double L = 1.0;                                // Lipschitz constant estimate

    // Main FISTA iteration loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        result.iterations = iter + 1;

        // Store previous values
        x_prev = x;
        t_prev = t;

        // Gradient step
        Eigen::VectorXd gradient = H_model.transpose() * (H_model * y - g_signal);
        x = y - (1.0/L) * gradient;

        // Soft thresholding (L1 regularization)
        double lambda = 0.1 / L;  // Regularization parameter
        x = (x.array().abs() > lambda).select(
            (x.array().abs() - lambda) * x.array().sign(),
            0.0
        );

        // Update t and y using FISTA update rule
        t = (1.0 + std::sqrt(1.0 + 4.0 * t * t)) / 2.0;
        y = x + ((t_prev - 1.0) / t) * (x - x_prev);

        // Calculate residual and update histories
        Eigen::VectorXd residual = H_model * x - g_signal;
        double current_residual_norm = residual.norm();
        result.residual_history.push_back(current_residual_norm);
        result.solution_history.push_back(x.norm());

        // Save intermediate result to CSV
        if (img_rows > 0 && img_cols > 0) {
            try {
                std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_" + std::to_string(iter) + ".csv");
                saveImageVectorToCsv(x, iter_img_path.string(), img_rows, img_cols);
            } catch (const std::exception& e) {
                std::cerr << "[AVISO] Falha ao salvar imagem iter " << iter << " (FISTA): " << e.what() << std::endl;
            }
        }

        // Check convergence
        if (iter > 0) {
            double epsilon = std::abs(result.residual_history[iter] - result.residual_history[iter-1]);
            result.final_epsilon = epsilon;
            if (epsilon < tolerance) {
                result.converged = true;
                break;
            }
        }
    }

    // Set final results
    result.image = x;
    result.final_error = (H_model * x - g_signal).norm();
    const auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}

// --- Instanciações explícitas para os tipos de matrizes ---
// Garante que o compilador gere o código para SparseMatrix<double>
template ReconstructionResult run_cgnr_solver_epsilon_save_iters<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_preconditioned_save_iters<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_fixed_iter<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    int num_iterations);

// **** DESCOMENTE ESTAS LINHAS ****
template ReconstructionResult run_cgnr_solver_epsilon_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

/* // Não estamos usando pré-condicionador denso, então esta pode ficar comentada se quiser
template ReconstructionResult run_cgnr_solver_preconditioned_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);
*/

template ReconstructionResult run_cgnr_solver_fixed_iter<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    int num_iterations);
// **** FIM DA CORREÇÃO ****

// Se você precisar rodar com Matriz Densa também, descomente estas:
/*
template ReconstructionResult run_cgnr_solver_epsilon_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_preconditioned_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_fixed_iter<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    int num_iterations);
*/

// Explicit template instantiation for FISTA solver
template ReconstructionResult run_fista_solver_save_iters<Eigen::SparseMatrix<double>>(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_fista_solver_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);

/* // Commented out dense matrix implementations
template ReconstructionResult run_fista_solver_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string& base_filename_prefix, const std::filesystem::path& output_dir,
    int img_rows, int img_cols);
*/
