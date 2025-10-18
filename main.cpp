#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <stdexcept>
#include <numeric>
#include <algorithm> // Para std::find

#include <Eigen/Dense>
#include <Eigen/Sparse>

// --- Estruturas de Dados ---

struct ReconstructionResult {
    Eigen::VectorXd image;
    int iterations;
    double final_error;
    double execution_time_ms;
    bool converged;
};

struct TestConfig {
    std::string test_name;
    std::string h_matrix_path;
    std::string g_signal_path;
    int image_rows; // Dimensão da imagem (e.g., 60 para 60x60)
    int image_cols; // Dimensão da imagem (e.g., 60 para 60x60)
    int Ne; // Número de elementos do transdutor (K na tese)
    int S;  // Número de amostras por elemento
};

struct PerformanceMetrics {
    double load_time_ms = 0.0;
    double solve_time_ms = 0.0;
    double estimated_ram_mb = 0.0;
    int iterations = 0;
    double final_error = 0.0;
    bool converged = false;
};

struct TestResult {
    std::string test_name;
    PerformanceMetrics dense_text;
    PerformanceMetrics dense_binary;
    PerformanceMetrics sparse_text;
    PerformanceMetrics sparse_binary;
    PerformanceMetrics symmetric_sparse_binary;
};

// --- Classe para Matriz Simétrica Virtual (CORRIGIDA) ---

class SymmetricHMatrix {
public:
    SymmetricHMatrix(const Eigen::SparseMatrix<double>& h_left, int img_rows, int img_cols, int Ne, int S)
        : H_left_(h_left),
          full_rows_(h_left.rows()),
          image_rows_(img_rows),
          image_cols_(img_cols),
          full_cols_(img_cols * img_rows),
          Ne_(Ne),
          S_(S) {
        if (image_cols_ % 2 != 0) {
            throw std::invalid_argument("O numero de colunas da imagem para a matriz simetrica deve ser par.");
        }
    }

    long long rows() const { return full_rows_; }
    long long cols() const { return full_cols_; }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const;
    Eigen::VectorXd transposeMultiply(const Eigen::VectorXd& x) const;

private:
    const Eigen::SparseMatrix<double>& H_left_;
    long long full_rows_;
    long long full_cols_;
    int image_rows_;
    int image_cols_;
    int Ne_;
    int S_;
};

// --- Funções de I/O (sem alterações, exceto a nova função) ---

Eigen::VectorXd loadVectorData(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<double> values;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            try { values.push_back(std::stod(cell)); } catch (const std::invalid_argument &) {}
        }
    }
    return Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
}

Eigen::MatrixXd loadDenseData(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<double> values;
    std::string line;
    int rows = 0;
    long long cols = 0;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        long long current_cols = 0;
        while (std::getline(lineStream, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
                current_cols++;
            } catch (const std::invalid_argument &) {}
        }
        if (rows == 0) cols = current_cols;
        rows++;
    }
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);
}

Eigen::SparseMatrix<double> convertCsvToSparse(const std::string &path, int image_size) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<Eigen::Triplet<double>> tripletList;
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',')) {
            try {
                if (double value = std::stod(cell); std::abs(value) > 1e-12) {
                    tripletList.emplace_back(row, col, value);
                }
            } catch (const std::invalid_argument&) {}
            col++;
        }
        row++;
    }
    Eigen::SparseMatrix<double> mat(row, image_size);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

void saveSparseMatrix(const Eigen::SparseMatrix<double> &mat, const std::string &path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel criar o arquivo binario: " + path);
    const auto rows = mat.rows(), cols = mat.cols(), nonZeros = mat.nonZeros();
    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char *>(&nonZeros), sizeof(nonZeros));
    file.write(reinterpret_cast<const char *>(mat.valuePtr()), nonZeros * sizeof(double));
    file.write(reinterpret_cast<const char *>(mat.innerIndexPtr()), nonZeros * sizeof(int));
    file.write(reinterpret_cast<const char *>(mat.outerIndexPtr()), (mat.outerSize() + 1) * sizeof(int));
}

Eigen::SparseMatrix<double> loadSparseMatrix(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " + path);
    Eigen::SparseMatrix<double>::Index rows, cols, nonZeros;
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char *>(&nonZeros), sizeof(nonZeros));
    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(nonZeros);
    file.read(reinterpret_cast<char *>(mat.valuePtr()), nonZeros * sizeof(double));
    file.read(reinterpret_cast<char *>(mat.innerIndexPtr()), nonZeros * sizeof(int));
    file.read(reinterpret_cast<char *>(mat.outerIndexPtr()), (mat.outerSize() + 1) * sizeof(int));
    mat.finalize();
    return mat;
}

void saveDenseMatrix(const Eigen::MatrixXd &mat, const std::string &path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel criar o arquivo binario: " + path);
    const auto rows = mat.rows(), cols = mat.cols();
    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char *>(mat.data()), rows * cols * sizeof(double));
}

Eigen::MatrixXd loadDenseMatrix(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " + path);
    Eigen::MatrixXd::Index rows, cols;
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    Eigen::MatrixXd mat(rows, cols);
    file.read(reinterpret_cast<char *>(mat.data()), rows * cols * sizeof(double));
    return mat;
}

Eigen::SparseMatrix<double> loadSymmetricSparseMatrix(const std::string& csv_path, const std::string& bin_path, int image_rows, int image_cols) {
    if (std::ifstream(bin_path).good()) {
        return loadSparseMatrix(bin_path);
    }
    std::cout << "[AVISO] Criando arquivo binario simetrico para " << csv_path << "..." << std::endl;
    std::ifstream file(csv_path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + csv_path);
    std::vector<Eigen::Triplet<double>> tripletList;
    std::string line;
    int row = 0;
    long long half_cols = (image_rows * image_cols) / 2;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',')) {
            if (col >= half_cols) break;
            try {
                if (double value = std::stod(cell); std::abs(value) > 1e-12) {
                    tripletList.emplace_back(row, col, value);
                }
            } catch (const std::invalid_argument&) {}
            col++;
        }
        row++;
    }
    Eigen::SparseMatrix<double> mat(row, half_cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    saveSparseMatrix(mat, bin_path);
    return mat;
}

// --- Implementação da Multiplicação Simétrica (CORRIGIDA) ---

Eigen::VectorXd SymmetricHMatrix::multiply(const Eigen::VectorXd& x) const {
    long long half_cols = full_cols_ / 2;
    Eigen::VectorXd y = H_left_ * x.head(half_cols);
    Eigen::VectorXd x_right_permuted(half_cols);

    #pragma omp parallel for
    for (int i = 0; i < image_cols_ / 2; ++i) {
        for (int j = 0; j < image_rows_; ++j) {
            int right_col_idx = (image_cols_ / 2 + i) * image_rows_ + j;
            int mirrored_col_idx = (image_cols_ / 2 - 1 - i) * image_rows_ + j;
            x_right_permuted(mirrored_col_idx) = x(right_col_idx);
        }
    }

    Eigen::VectorXd y2_intermediate = H_left_ * x_right_permuted;
    Eigen::VectorXd y2_final(full_rows_);

    #pragma omp parallel for
    for (long long k = 0; k < Ne_; ++k) {
        long long mirrored_k = (Ne_ - 1) - k;
        y2_final.segment(k * S_, S_) = y2_intermediate.segment(mirrored_k * S_, S_);
    }

    y += y2_final;
    return y;
}

Eigen::VectorXd SymmetricHMatrix::transposeMultiply(const Eigen::VectorXd& r) const {
    long long half_cols = full_cols_ / 2;
    Eigen::VectorXd z = Eigen::VectorXd::Zero(full_cols_);
    z.head(half_cols) = H_left_.transpose() * r;
    Eigen::VectorXd r_permuted(full_rows_);

    #pragma omp parallel for
    for (long long k = 0; k < Ne_; ++k) {
        long long mirrored_k = (Ne_ - 1) - k;
        r_permuted.segment(k * S_, S_) = r.segment(mirrored_k * S_, S_);
    }

    Eigen::VectorXd z_right_intermediate = H_left_.transpose() * r_permuted;

    #pragma omp parallel for
    for (int i = 0; i < image_cols_ / 2; ++i) {
        for (int j = 0; j < image_rows_; ++j) {
            int right_col_idx = (image_cols_ / 2 + i) * image_rows_ + j;
            int mirrored_col_idx = (image_cols_ / 2 - 1 - i) * image_rows_ + j;
            z(right_col_idx) = z_right_intermediate(mirrored_col_idx);
        }
    }
    return z;
}

// --- Algoritmos de Reconstrução (com sobrecarga para SymmetricHMatrix) ---

template<typename MatrixType>
ReconstructionResult run_cgnr_solver(const Eigen::VectorXd &g_signal, const MatrixType &H_model,
                                     const double tolerance = 1e-4, const int max_iterations = 100) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd r = g_signal;
    Eigen::VectorXd z = H_model.transpose() * r;
    Eigen::VectorXd p = z;
    double z_norm_sq = z.squaredNorm();
    ReconstructionResult result;
    result.iterations = 0;
    result.converged = false;
    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;
        Eigen::VectorXd w = H_model * p;
        double alpha = z_norm_sq / w.squaredNorm();
        f += alpha * p;
        r -= alpha * w;
        if (r.norm() < tolerance) {
            result.converged = true;
            break;
        }
        Eigen::VectorXd z_next = H_model.transpose() * r;
        const double z_next_norm_sq = z_next.squaredNorm();
        if (z_norm_sq == 0) break;
        double beta = z_next_norm_sq / z_norm_sq;
        p = z_next + beta * p;
        z = z_next;
        z_norm_sq = z_next_norm_sq;
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f;
    result.final_error = r.norm();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

ReconstructionResult run_cgnr_solver(const Eigen::VectorXd &g_signal, const SymmetricHMatrix &H_model,
                                     const double tolerance = 1e-4, const int max_iterations = 100) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd r = g_signal;
    Eigen::VectorXd z = H_model.transposeMultiply(r);
    Eigen::VectorXd p = z;
    double z_norm_sq = z.squaredNorm();
    ReconstructionResult result;
    result.iterations = 0;
    result.converged = false;
    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;
        Eigen::VectorXd w = H_model.multiply(p);
        double alpha = z_norm_sq / w.squaredNorm();
        f += alpha * p;
        r -= alpha * w;
        if (r.norm() < tolerance) {
            result.converged = true;
            break;
        }
        Eigen::VectorXd z_next = H_model.transposeMultiply(r);
        const double z_next_norm_sq = z_next.squaredNorm();
        if (z_norm_sq == 0) break;
        double beta = z_next_norm_sq / z_norm_sq;
        p = z_next + beta * p;
        z = z_next;
        z_norm_sq = z_next_norm_sq;
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f;
    result.final_error = r.norm();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

// --- Função Principal ---
int main(int argc, char* argv[]) {
    std::cout << "======================================================" << std::endl;
    std::cout << " Comparativo de Desempenho: Denso vs. Esparso vs. Simetrico" << std::endl;
    std::cout << "======================================================" << std::endl;

    std::vector<std::string> args(argv + 1, argv + argc);
    bool run_dense = false;
    bool run_sparse = false;
    bool run_symmetric = false;

    if (args.empty() || std::find(args.begin(), args.end(), "--all") != args.end()) {
        run_dense = run_sparse = run_symmetric = true;
    } else {
        for (const auto& arg : args) {
            if (arg == "--dense") run_dense = true;
            if (arg == "--sparse") run_sparse = true;
            if (arg == "--symmetric") run_symmetric = true;
        }
    }

    std::vector<TestConfig> tests = {
        {"60x60 (Sinal G-1)", "../data/H-1.csv", "../data/G-1.csv", 60, 60, 64, 440},
        {"60x60 (Sinal G-2)", "../data/H-1.csv", "../data/G-2.csv", 60, 60, 64, 440},
        {"30x30 (Sinal g-1)", "../data/H-2.csv", "../data/g-30x30-1.csv", 30, 30, 64, 220},
        {"30x30 (Sinal g-2)", "../data/H-2.csv", "../data/g-30x30-2.csv", 30, 30, 64, 220}
    };

    Eigen::setNbThreads(omp_get_max_threads());
    std::cout << "\n[INFO] Usando " << Eigen::nbThreads() << " threads para os calculos.\n" << std::endl;

    std::cout << "[INFO] Verificando/Criando arquivos binarios para acelerar a leitura..." << std::endl;
    for (const auto &config: tests) {
        int image_size = config.image_rows * config.image_cols;
        if (run_sparse) {
            std::string sparse_bin_path = config.h_matrix_path + ".sparse.bin";
            if (!std::ifstream(sparse_bin_path).good()) {
                std::cout << "[AVISO] Criando arquivo binario esparso para " << config.h_matrix_path << "..." << std::endl;
                saveSparseMatrix(convertCsvToSparse(config.h_matrix_path, image_size), sparse_bin_path);
            }
        }
        if (run_dense) {
            std::string dense_bin_path = config.h_matrix_path + ".dense.bin";
            if (!std::ifstream(dense_bin_path).good()) {
                std::cout << "[AVISO] Criando arquivo binario denso para " << config.h_matrix_path << "..." << std::endl;
                saveDenseMatrix(loadDenseData(config.h_matrix_path), dense_bin_path);
            }
        }
        if (run_symmetric) {
            std::string symm_bin_path = config.h_matrix_path + ".symm.sparse.bin";
            if (!std::ifstream(symm_bin_path).good()) {
                 loadSymmetricSparseMatrix(config.h_matrix_path, symm_bin_path, config.image_rows, config.image_cols);
            }
        }
    }
    std::cout << "[INFO] Pre-processamento concluido.\n" << std::endl;

    std::vector<TestResult> all_results;

    for (const auto &config: tests) {
        TestResult current_test;
        current_test.test_name = config.test_name;
        int image_size = config.image_rows * config.image_cols;
        std::cout << "\n========================================\nINICIANDO TESTE: " << config.test_name <<
                "\n========================================" << std::endl;

        if (run_dense) {
            try {
                std::cout << "\n--- 1. Denso de Texto ---\n";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::MatrixXd H = loadDenseData(config.h_matrix_path);
                Eigen::VectorXd g = loadVectorData(config.g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.dense_text.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                current_test.dense_text.estimated_ram_mb = static_cast<double>(H.rows()) * H.cols() * sizeof(double) / (1024 * 1024);
                ReconstructionResult res = run_cgnr_solver(g, H);
                current_test.dense_text.solve_time_ms = res.execution_time_ms;
                current_test.dense_text.iterations = res.iterations;
                current_test.dense_text.final_error = res.final_error;
                current_test.dense_text.converged = res.converged;
            } catch (const std::exception &e) { std::cerr << "[ERRO] " << e.what() << std::endl; }

            try {
                std::cout << "\n--- 2. Denso de Binario ---\n";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::MatrixXd H = loadDenseMatrix(config.h_matrix_path + ".dense.bin");
                Eigen::VectorXd g = loadVectorData(config.g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.dense_binary.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                current_test.dense_binary.estimated_ram_mb = static_cast<double>(H.rows()) * H.cols() * sizeof(double) / (1024 * 1024);
                ReconstructionResult res = run_cgnr_solver(g, H);
                current_test.dense_binary.solve_time_ms = res.execution_time_ms;
                current_test.dense_binary.iterations = res.iterations;
                current_test.dense_binary.final_error = res.final_error;
                current_test.dense_binary.converged = res.converged;
            } catch (const std::exception &e) { std::cerr << "[ERRO] " << e.what() << std::endl; }
        }

        if (run_sparse) {
            try {
                std::cout << "\n--- 3. Esparso de Texto (Conversao) ---\n";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::SparseMatrix<double> H = convertCsvToSparse(config.h_matrix_path, image_size);
                Eigen::VectorXd g = loadVectorData(config.g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.sparse_text.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                current_test.sparse_text.estimated_ram_mb = static_cast<double>(H.nonZeros() * (sizeof(double) + sizeof(int)) + H.outerSize() * sizeof(int)) / (1024 * 1024);
                ReconstructionResult res = run_cgnr_solver(g, H);
                current_test.sparse_text.solve_time_ms = res.execution_time_ms;
                current_test.sparse_text.iterations = res.iterations;
                current_test.sparse_text.final_error = res.final_error;
                current_test.sparse_text.converged = res.converged;
            } catch (const std::exception &e) { std::cerr << "[ERRO] " << e.what() << std::endl; }

            try {
                std::cout << "\n--- 4. Esparso de Binario ---\n";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::SparseMatrix<double> H = loadSparseMatrix(config.h_matrix_path + ".sparse.bin");
                Eigen::VectorXd g = loadVectorData(config.g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.sparse_binary.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                current_test.sparse_binary.estimated_ram_mb = static_cast<double>(H.nonZeros() * (sizeof(double) + sizeof(int)) + H.outerSize() * sizeof(int)) / (1024 * 1024);
                ReconstructionResult res = run_cgnr_solver(g, H);
                current_test.sparse_binary.solve_time_ms = res.execution_time_ms;
                current_test.sparse_binary.iterations = res.iterations;
                current_test.sparse_binary.final_error = res.final_error;
                current_test.sparse_binary.converged = res.converged;
            } catch (const std::exception &e) { std::cerr << "[ERRO] " << e.what() << std::endl; }
        }

        if (run_symmetric) {
            try {
                std::cout << "\n--- 5. Simetrico Esparso de Binario ---\n";
                std::string bin_path = config.h_matrix_path + ".symm.sparse.bin";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::SparseMatrix<double> H_left = loadSymmetricSparseMatrix(config.h_matrix_path, bin_path, config.image_rows, config.image_cols);
                SymmetricHMatrix H_symm(H_left, config.image_rows, config.image_cols, config.Ne, config.S);
                Eigen::VectorXd g = loadVectorData(config.g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.symmetric_sparse_binary.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                current_test.symmetric_sparse_binary.estimated_ram_mb = static_cast<double>(H_left.nonZeros() * (sizeof(double) + sizeof(int)) + H_left.outerSize() * sizeof(int)) / (1024 * 1024);
                ReconstructionResult res = run_cgnr_solver(g, H_symm);
                current_test.symmetric_sparse_binary.solve_time_ms = res.execution_time_ms;
                current_test.symmetric_sparse_binary.iterations = res.iterations;
                current_test.symmetric_sparse_binary.final_error = res.final_error;
                current_test.symmetric_sparse_binary.converged = res.converged;
            } catch (const std::exception &e) { std::cerr << "[ERRO] " << e.what() << std::endl; }
        }

        all_results.push_back(current_test);
    }

    // --- Tabela Final de Resultados ---
    std::cout << "\n\n=================================================================================================================================" << std::endl;
    std::cout << "                                           RELATORIO COMPARATIVO FINAL" << std::endl;
    std::cout << "=================================================================================================================================" << std::endl;
    std::cout << std::left
              << std::setw(22) << "Teste"
              << std::setw(26) << "Metodo"
              << std::setw(15) << "RAM (MB)"
              << std::setw(20) << "T. Carga (ms)"
              << std::setw(20) << "T. Solver (ms)"
              << std::setw(12) << "Iteracoes"
              << std::setw(15) << "Erro Final"
              << std::setw(15) << "vs Baseline"
              << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto &res : all_results) {
        const auto& baseline = res.sparse_binary;

        auto print_metric = [&](const std::string &label, const PerformanceMetrics &m, bool is_baseline) {
            if (m.load_time_ms == 0 && m.solve_time_ms == 0) return;

            std::cout << std::left << std::setw(22) << (label == "Denso / Texto" ? res.test_name : "")
                      << std::setw(26) << label
                      << std::fixed << std::setprecision(2) << std::setw(15) << m.estimated_ram_mb
                      << std::setw(20) << m.load_time_ms
                      << std::setw(20) << m.solve_time_ms
                      << std::setw(12) << m.iterations
                      << std::scientific << std::setprecision(3) << m.final_error;

            if (is_baseline) {
                std::cout << std::setw(15) << "BASELINE";
            } else if (baseline.solve_time_ms > 0) {
                double speedup = baseline.solve_time_ms / m.solve_time_ms;
                std::cout << std::fixed << std::setprecision(2) << std::setw(14) << std::right << (std::to_string(speedup) + "x");
            }
            std::cout << std::endl;
        };

        print_metric("Denso / Texto", res.dense_text, false);
        print_metric("Denso / Binario", res.dense_binary, false);
        print_metric("Esparso / Texto", res.sparse_text, false);
        print_metric("Esparso / Binario", res.sparse_binary, true);
        print_metric("Simetrico / Binario", res.symmetric_sparse_binary, false);

        if (res.sparse_binary.estimated_ram_mb > 0 && res.symmetric_sparse_binary.estimated_ram_mb > 0) {
             double mem_reduction = 100.0 * (1.0 - res.symmetric_sparse_binary.estimated_ram_mb / res.sparse_binary.estimated_ram_mb);
             std::cout << std::left << std::setw(22) << ""
                       << std::setw(26) << "Ganho (Simetrico vs Esparso)"
                       << std::fixed << std::setprecision(1)
                       << "Reducao " << mem_reduction << "% RAM"
                       << std::endl;
        }
        std::cout << "---------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    }

    return 0;
}