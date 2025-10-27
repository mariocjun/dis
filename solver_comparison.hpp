#ifndef ULTRASOUNDSERVER_SOLVER_COMPARISON_HPP
#define ULTRASOUNDSERVER_SOLVER_COMPARISON_HPP

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
#include <algorithm>
#include <limits>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

// Inclui as definições de structs do seu próprio projeto
#include "include/types.hpp"

// --- Funções Auxiliares (Implementações Completas) ---
// (As 11 funções: loadVectorData, convertCsvToSparse, saveSparseMatrix, loadSparseMatrix,
// loadDenseData, loadDenseMatrix, saveDenseMatrix, saveHistoryToCSV, saveLcurveToCSV,
// saveImageVectorToCsv, normalize_system_rows... ESTÃO AQUI)

inline Eigen::VectorXd loadVectorData(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<double> values;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || std::ranges::all_of(line, ::isspace)) continue;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            try {
                std::erase_if(cell, ::isspace);
                if (!cell.empty()) {
                    values.push_back(std::stod(cell));
                }
            } catch (const std::invalid_argument &) {
                std::cerr << "[AVISO] Ignorando valor nao numerico em: " << path << ", linha: " << line_num <<
                        ", celula: '" << cell << "'" << std::endl;
            } catch (const std::out_of_range &) {
                std::cerr << "[AVISO] Ignorando valor fora do range em: " << path << ", linha: " << line_num <<
                        ", celula: '" << cell << "'" << std::endl;
            }
        }
    }
    if (values.empty()) {
        throw std::runtime_error("Nenhum dado numerico valido encontrado em: " + path);
    }
    Eigen::Map<Eigen::VectorXd> vec_map(values.data(), values.size());
    return Eigen::VectorXd(vec_map);
}

inline Eigen::SparseMatrix<double> convertCsvToSparse(const std::string &path, int expected_cols) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<Eigen::Triplet<double> > tripletList;
    std::string line;
    int row = 0;
    long long actual_cols = -1;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || std::ranges::all_of(line, ::isspace)) continue;

        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',')) {
            try {
                std::erase_if(cell, ::isspace);
                if (!cell.empty()) {
                    double value = std::stod(cell);
                    if (std::abs(value) > 1e-12) {
                        tripletList.emplace_back(row, col, value);
                    }
                }
            } catch (const std::invalid_argument &) {
                std::cerr << "[AVISO] Ignorando valor nao numerico em CSV esparso: " << path << ", linha: " << line_num
                        << ", celula: '" << cell << "'" << std::endl;
            } catch (const std::out_of_range &) {
                std::cerr << "[AVISO] Ignorando valor fora do range em CSV esparso: " << path << ", linha: " << line_num
                        << ", celula: '" << cell << "'" << std::endl;
            }
            col++;
        }
        if (actual_cols == -1) {
            if (col == 0) continue;
            actual_cols = col;
        } else if (col != actual_cols) {
            std::cerr << "[ERRO] Numero inconsistente de colunas na linha " << line_num << " do arquivo esparso " <<
                    path << ". Esperado: " << actual_cols << ", Encontrado: " << col << ". Abortando." << std::endl;
            throw std::runtime_error("Inconsistencia de colunas no CSV esparso.");
        }
        row++;
    }
    if (row == 0 || actual_cols <= 0) {
        throw std::runtime_error("Nao foi possivel ler nenhuma linha/coluna valida do arquivo esparso: " + path);
    }
    if (expected_cols > 0 && actual_cols != expected_cols) {
        std::cerr << "[AVISO] Numero de colunas lido (" << actual_cols << ") difere do esperado (" << expected_cols <<
                ") para " << path << ". Usando o numero lido." << std::endl;
    }

    Eigen::SparseMatrix<double> mat(row, actual_cols);
    if (!tripletList.empty()) {
        mat.setFromTriplets(tripletList.begin(), tripletList.end());
    } else {
        std::cerr << "[AVISO] Nenhum elemento nao-zero (acima de 1e-12) encontrado em " << path << std::endl;
    }
    mat.makeCompressed();
    return mat;
}

inline void saveSparseMatrix(const Eigen::SparseMatrix<double> &mat, const std::string &path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel criar o arquivo binario: " + path);

    Eigen::SparseMatrix<double> compressed_mat = mat;
    if (!compressed_mat.isCompressed()) {
        compressed_mat.makeCompressed();
    }

    const auto rows = compressed_mat.rows();
    const auto cols = compressed_mat.cols();
    const auto nonZeros = compressed_mat.nonZeros();
    const auto outerSize = compressed_mat.outerSize();

    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char *>(&nonZeros), sizeof(nonZeros));

    if (nonZeros > 0) {
        if (!compressed_mat.valuePtr() || !compressed_mat.innerIndexPtr()) {
            throw std::runtime_error("Ponteiros internos invalidos ao salvar matriz esparsa para: " + path);
        }
        file.write(reinterpret_cast<const char *>(compressed_mat.valuePtr()), nonZeros * sizeof(double));
        file.write(reinterpret_cast<const char *>(compressed_mat.innerIndexPtr()), nonZeros * sizeof(int));
    }
    if (!compressed_mat.outerIndexPtr()) {
        throw std::runtime_error("Ponteiro outerIndex invalido ao salvar matriz esparsa para: " + path);
    }
    file.write(reinterpret_cast<const char *>(compressed_mat.outerIndexPtr()), (outerSize + 1) * sizeof(int));

    if (!file) throw std::runtime_error("Erro ao escrever no arquivo binario esparso: " + path);
    file.close();
    if (!file) throw std::runtime_error("Erro ao fechar o arquivo binario esparso: " + path);
}


inline Eigen::SparseMatrix<double> loadSparseMatrix(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " + path);
    Eigen::Index rows, cols;
    Eigen::Index nonZeros;
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char *>(&nonZeros), sizeof(nonZeros));

    if (!file || file.eof()) throw std::runtime_error("Erro ao ler cabecalho do arquivo binario esparso: " + path);
    if (rows < 0 || cols < 0 || nonZeros < 0) throw std::runtime_error(
        "Dimensoes invalidas (" + std::to_string(rows) + "x" + std::to_string(cols) + ", nnz=" +
        std::to_string(nonZeros) + ") lidas do arquivo binario esparso: " + path);

    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(nonZeros);

    if (nonZeros > 0) {
        if (!mat.valuePtr() || !mat.innerIndexPtr()) {
            throw std::runtime_error("Ponteiros value/inner invalidos apos resizeNonZeros ao carregar: " + path);
        }
        file.read(reinterpret_cast<char *>(mat.valuePtr()), nonZeros * sizeof(double));
        file.read(reinterpret_cast<char *>(mat.innerIndexPtr()), nonZeros * sizeof(int));
    }

    if (!mat.outerIndexPtr()) {
        throw std::runtime_error("Ponteiro outerIndex invalido apos makeCompressed ao carregar: " + path);
    }
    file.read(reinterpret_cast<char *>(mat.outerIndexPtr()), (mat.outerSize() + 1) * sizeof(int));

    if (!file) {
        if (file.eof()) {
            std::cerr << "[AVISO] Fim de arquivo prematuro ao ler dados de " << path <<
                    ". A matriz pode estar incompleta." << std::endl;
            throw std::runtime_error("Fim de arquivo inesperado ao ler dados do arquivo binario esparso: " + path);
        } else {
            throw std::runtime_error("Erro de leitura ao processar dados do arquivo binario esparso: " + path);
        }
    }
    mat.finalize();
    return mat;
}

inline Eigen::MatrixXd loadDenseData(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<double> values;
    std::string line;
    int rows = 0;
    long long cols = -1;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) continue;

        std::stringstream lineStream(line);
        std::string cell;
        long long current_cols = 0;
        std::vector<double> row_values;

        while (std::getline(lineStream, cell, ',')) {
            try {
                cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
                if (!cell.empty()) {
                    row_values.push_back(std::stod(cell));
                } else {
                    row_values.push_back(0.0);
                    std::cerr << "[AVISO] Celula vazia encontrada em: " << path << ", linha: " << line_num <<
                            ", coluna: " << current_cols + 1 << ". Assumindo 0.0." << std::endl;
                }
                current_cols++;
            } catch (const std::invalid_argument &) {
                std::cerr << "[AVISO] Ignorando valor nao numerico em: " << path << ", linha: " << line_num <<
                        ", celula: '" << cell << "'" << std::endl;
                row_values.push_back(0.0);
                current_cols++;
            } catch (const std::out_of_range &) {
                std::cerr << "[AVISO] Ignorando valor fora do range em: " << path << ", linha: " << line_num <<
                        ", celula: '" << cell << "'" << std::endl;
                row_values.push_back(0.0);
                current_cols++;
            }
        }

        if (cols == -1) {
            if (current_cols == 0) continue;
            cols = current_cols;
        } else if (current_cols != cols) {
            std::cerr << "[ERRO] Numero inconsistente de colunas na linha " << line_num << " do arquivo " << path <<
                    ". Esperado: " << cols << ", Encontrado: " << current_cols << ". Abortando." << std::endl;
            throw std::runtime_error("Inconsistencia de colunas no CSV denso.");
        }

        values.insert(values.end(), row_values.begin(), row_values.end());
        rows++;
    }
    if (rows == 0 || cols <= 0 || values.empty()) {
        throw std::runtime_error(
            "Nao foi possivel carregar dados validos da matriz densa de: " + path + " (rows=" + std::to_string(rows) +
            ", cols=" + std::to_string(cols) + ")");
    }

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > mat_map(
        values.data(), rows, cols);
    return Eigen::MatrixXd(mat_map);
}

inline Eigen::MatrixXd loadDenseMatrix(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " + path);
    Eigen::Index rows, cols;
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    if (!file || file.eof()) throw std::runtime_error("Erro ao ler cabecalho do arquivo binario denso: " + path);
    if (rows <= 0 || cols <= 0) throw std::runtime_error(
        "Dimensoes invalidas (" + std::to_string(rows) + "x" + std::to_string(cols) +
        ") lidas do arquivo binario denso: " + path);

    Eigen::MatrixXd mat(rows, cols);
    file.read(reinterpret_cast<char *>(mat.data()), static_cast<std::streamsize>(rows) * cols * sizeof(double));
    if (!file) {
        if (file.eof() && (static_cast<std::streamsize>(rows) * cols * sizeof(double) > 0)) {
            throw std::runtime_error("Fim de arquivo inesperado ao ler dados do arquivo binario denso: " + path);
        } else if (!file.eof()) {
            throw std::runtime_error("Erro de leitura ao processar dados do arquivo binario denso: " + path);
        }
    }
    return mat;
}

inline void saveDenseMatrix(const Eigen::MatrixXd &mat, const std::string &path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel criar o arquivo binario: " + path);
    const auto rows = mat.rows(), cols = mat.cols();
    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char *>(mat.data()), static_cast<std::streamsize>(rows) * cols * sizeof(double));
    if (!file) throw std::runtime_error("Erro ao escrever no arquivo binario denso: " + path);
    file.close();
    if (!file) throw std::runtime_error("Erro ao fechar o arquivo binario denso: " + path);
}


inline void saveHistoryToCSV(const std::vector<double> &history, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[AVISO] Nao foi possivel criar o arquivo de historico: " << filename << std::endl;
        return;
    }
    file << "Iteration,ResidualNorm\n";
    for (size_t i = 0; i < history.size(); ++i) {
        file << i + 1 << "," << std::scientific << std::setprecision(8) << history[i] << "\n";
    }
    if (!file) {
        std::cerr << "[AVISO] Erro ao escrever no arquivo de historico: " << filename << std::endl;
    }
    file.close();
}

// Função auxiliar para manter compatibilidade com ReconstructionResult
inline void saveHistoryToCSV(const ReconstructionResult &result, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[AVISO] Nao foi possivel criar o arquivo de historico: " << filename << std::endl;
        return;
    }

    // Cabeçalho com mais métricas
    file << "Iteration,ResidualNorm,SolutionNorm,ExecutionTime_ms\n";

    for (size_t i = 0; i < result.residual_history.size(); ++i) {
        file << i + 1 << ","
                << std::scientific << std::setprecision(8) << result.residual_history[i] << ","
                << std::scientific << std::setprecision(8) << (i < result.solution_history.size()
                                                                   ? result.solution_history[i]
                                                                   : 0.0) << ","
                << std::fixed << std::setprecision(2) << result.execution_time_ms << "\n";
    }

    if (!file) {
        std::cerr << "[AVISO] Erro ao escrever no arquivo de historico: " << filename << std::endl;
    } else {
        std::cout << "[INFO] Historico de convergencia salvo em: " << filename << std::endl;
    }
    file.close();
}

inline void saveLcurveToCSV(const ReconstructionResult &result, const std::string &filename) {
    if (result.residual_history.size() != result.solution_history.size()) {
        std::cerr << "[AVISO] Tamanhos incompativeis (" << result.residual_history.size() << " vs "
                << result.solution_history.size() << ") de historico de residuo e solucao para L-curve. Nao salvando "
                << filename << std::endl;
        return;
    }
    if (result.residual_history.empty()) {
        std::cerr << "[AVISO] Historico vazio, nada para salvar em L-curve: " << filename << std::endl;
        return;
    }
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[AVISO] Nao foi possivel criar o arquivo de cotovelo: " << filename << std::endl;
        return;
    }
    file << "Iteration,SolutionNorm,ResidualNorm\n";
    for (size_t i = 0; i < result.residual_history.size(); ++i) {
        file << i + 1 << ","
                << std::scientific << std::setprecision(8) << result.solution_history[i] << ","
                << std::scientific << std::setprecision(8) << result.residual_history[i] << "\n";
    }
    if (!file) {
        std::cerr << "[AVISO] Erro ao escrever no arquivo L-curve: " << filename << std::endl;
    } else {
        std::cout << "[INFO] Dados L-curve salvos em: " << filename << std::endl;
    }
    file.close();
}

inline void saveImageVectorToCsv(const Eigen::VectorXd &vec, const std::string &filename, int img_rows, int img_cols) {
    if (vec.size() != static_cast<long long>(img_rows) * img_cols) {
        std::cerr << "[AVISO] Tamanho do vetor (" << vec.size() << ") nao corresponde as dimensoes da imagem ("
                << img_rows << "x" << img_cols << "). Nao salvando imagem: " << filename << std::endl;
        return;
    }
    if (img_rows <= 0 || img_cols <= 0) {
        std::cerr << "[AVISO] Dimensoes invalidas da imagem (" << img_rows << "x" << img_cols
                << "). Nao salvando imagem: " << filename << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[AVISO] Nao foi possivel criar o arquivo CSV da imagem: " << filename << std::endl;
        return;
    }

    file << std::scientific << std::setprecision(8);

    for (int i = 0; i < img_rows; ++i) {
        for (int j = 0; j < img_cols; ++j) {
            long long index = static_cast<long long>(j) * img_rows + i; // ColMajor Indexing
            if (index < vec.size()) {
                file << vec(index);
            } else {
                file << 0.0;
            }
            if (j < img_cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    if (!file) {
        std::cerr << "[AVISO] Erro ao escrever no arquivo CSV da imagem: " << filename << std::endl;
    } else {
        std::cout << "[INFO] Imagem reconstruida salva em: " << filename << std::endl;
    }
    file.close();
}

template<typename MatrixType>
inline void normalize_system_rows(MatrixType &H, Eigen::VectorXd &g) {
    if (H.rows() != g.size()) {
        throw std::runtime_error("normalize_system_rows: Dimensoes H/g incompativeis.");
    }
    if (H.rows() == 0) return;

    std::cout << "[INFO] Normalizando linhas de H e elementos de g..." << std::endl;
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
        std::vector<Eigen::Triplet<double> > triplets_normalized;
        triplets_normalized.reserve(H.nonZeros());
        for (int k = 0; k < H.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(H, k); it; ++it) {
                double row_norm = std::sqrt(row_norms_sq[it.row()]);
                if (row_norm > epsilon_norm) {
                    triplets_normalized.emplace_back(it.row(), it.col(), it.value() / row_norm);
                }
            }
        }
        H.setFromTriplets(triplets_normalized.begin(), triplets_normalized.end());
#pragma omp parallel for
        for (Eigen::Index i = 0; i < g.size(); ++i) {
            double row_norm = std::sqrt(row_norms_sq[i]);
            if (row_norm > epsilon_norm) { g(i) /= row_norm; } else { g(i) = 0.0; }
        }
    }
    std::cout << "[INFO] Normalizacao concluida." << std::endl;
}


// --- Solver CGNR Regularizado (Salva Imagens Intermediárias) ---
template<typename MatrixType>
inline ReconstructionResult run_cgnr_solver_epsilon_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string &base_filename_prefix,
    const std::filesystem::path &output_dir,
    int img_rows, int img_cols) {
    if (H_model.rows() != g_signal.size()) throw std::runtime_error(
        "Dimensoes incompativeis: H.rows()!=" + std::to_string(H_model.rows()) + " vs g.size()=" + std::to_string(
            g_signal.size()));
    if (H_model.cols() <= 0) throw std::runtime_error("Matriz H tem " + std::to_string(H_model.cols()) + " colunas.");
    if (H_model.rows() == 0) return ReconstructionResult{};


    const auto start_time = std::chrono::high_resolution_clock::now();

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
            std::cout << "[INFO] Lambda calculado era quase zero, usando piso minimo: " << lambda << std::endl;
        }
    } else {
        lambda = 1e-9;
        std::cout << "[AVISO] Vetor z inicial vazio, usando lambda=" << lambda << " como fallback." << std::endl;
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

    bool save_iters = !base_filename_prefix.empty() && img_rows > 0 && img_cols > 0;
    if (save_iters) {
        try {
            std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_0.csv");
            saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
        } catch (const std::exception &e) {
            std::cerr << "[AVISO] Falha ao salvar imagem iter 0: " << e.what() << std::endl;
        }
    }


    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;
        Eigen::VectorXd w = H_model * p;
        double p_norm_sq = p.squaredNorm();
        double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

        if (modified_denominator < std::numeric_limits<double>::epsilon()) {
            std::cout << "[INFO] Denominador modificado (...) proximo de zero na iteracao " << i + 1 << ". Parando." <<
                    std::endl;
            break;
        }

        double alpha = z_norm_sq / modified_denominator;
        f += alpha * p;
        r -= alpha * w;

        if (save_iters) {
            try {
                std::filesystem::path iter_img_path =
                        output_dir / (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
                saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
            } catch (const std::exception &e) {
                std::cerr << "[AVISO] Falha ao salvar imagem iter " << i + 1 << ": " << e.what() << std::endl;
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
            std::cout << "[INFO] Convergencia por epsilon atingida na iteracao " << i + 1 << " (epsilon=" <<
                    std::scientific << epsilon << " < " << tolerance << ")" << std::defaultfloat << std::endl;
            break;
        }
        previous_residual_norm = current_residual_norm;

        double beta = 0.0;
        if (z_norm_sq >= std::numeric_limits<double>::epsilon()) { beta = z_next_norm_sq / z_norm_sq; } else {
            std::cout << "[INFO] ||z||^2 (...) proximo de zero na iteracao " << i + 1 << ". Usando beta=0 (restart)." <<
                    std::endl;
            if (z_next_norm_sq < std::numeric_limits<double>::epsilon()) {
                std::cout << "[INFO] ||z_next||^2 tambem proximo de zero. Provavelmente estagnou. Parando." <<
                        std::endl;
                break;
            }
        }

        p = z_next + beta * p;
        z = z_next;
        z_norm_sq = z_next_norm_sq;

        if (i == max_iterations - 1 && !result.converged) {
            std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations <<
                    ") atingido sem convergencia por epsilon (ultimo epsilon=" << std::scientific << epsilon <<
                    std::defaultfloat << ")." << std::endl;
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f;
    result.final_error = current_residual_norm;
    result.final_epsilon = epsilon;
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

// --- Solver CGNR Pré-condicionado (Salva Imagens Intermediárias) ---
template<typename MatrixType>
inline ReconstructionResult run_cgnr_solver_preconditioned_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string &base_filename_prefix,
    const std::filesystem::path &output_dir,
    int img_rows, int img_cols) {
    if (H_model.rows() != g_signal.size()) throw std::runtime_error("...");
    if (H_model.cols() <= 0) throw std::runtime_error("...");
    if (H_model.rows() == 0) return ReconstructionResult{};

    const auto start_time = std::chrono::high_resolution_clock::now();

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
            std::cout << "[INFO] Lambda calculado era quase zero, usando piso minimo: " << lambda << std::endl;
        }
    } else {
        lambda = 1e-9;
        std::cout << "[AVISO] Vetor z inicial (unprec) vazio, usando lambda=" << lambda << " como fallback." <<
                std::endl;
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

    if (!base_filename_prefix.empty() && img_rows > 0 && img_cols > 0) {
        try {
            std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_0.csv");
            saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
        } catch (const std::exception &e) {
            std::cerr << "[AVISO] Falha ao salvar imagem iter 0 (precond): " << e.what() << std::endl;
        }
    }

    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;
        Eigen::VectorXd w = H_model * p;
        double p_norm_sq = p.squaredNorm();
        double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

        if (modified_denominator < std::numeric_limits<double>::epsilon()) {
            std::cout << "[INFO] Denominador modificado (...) proximo de zero na iteracao " << i + 1 << ". Parando." <<
                    std::endl;
            break;
        }

        double alpha = z_precond_dot_z / modified_denominator;
        f += alpha * p;
        r -= alpha * w;

        if (!base_filename_prefix.empty() && img_rows > 0 && img_cols > 0) {
            try {
                std::filesystem::path iter_img_path =
                        output_dir / (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
                saveImageVectorToCsv(f, iter_img_path.string(), img_rows, img_cols);
            } catch (const std::exception &e) {
                std::cerr << "[AVISO] Falha ao salvar imagem iter " << i + 1 << " (precond): " << e.what() << std::endl;
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
            std::cout << "[INFO] Convergencia por epsilon atingida na iteracao " << i + 1 << " (epsilon=" <<
                    std::scientific << epsilon << " < " << tolerance << ")" << std::defaultfloat << std::endl;
            break;
        }
        previous_residual_norm = current_residual_norm;

        double beta = 0.0;
        if (z_precond_dot_z >= std::numeric_limits<double>::epsilon()) {
            beta = z_next_precond_dot_z_next / z_precond_dot_z;
        } else {
            std::cout << "[INFO] ||z||^2 (...) proximo de zero na iteracao " << i + 1 << ". Usando beta=0 (restart)." <<
                    std::endl;
            if (z_next_precond_dot_z_next < std::numeric_limits<double>::epsilon()) {
                std::cout << "[INFO] ||z_next||^2 tambem proximo de zero. Provavelmente estagnou. Parando." <<
                        std::endl;
                break;
            }
        }

        p = z_next + beta * p;
        z = z_next;
        z_precond_dot_z = z_next_precond_dot_z_next;

        if (i == max_iterations - 1 && !result.converged) {
            std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations <<
                    ") atingido sem convergencia por epsilon (ultimo epsilon=" << std::scientific << epsilon <<
                    std::defaultfloat << ")." << std::endl;
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f;
    result.final_error = current_residual_norm;
    result.final_epsilon = epsilon;
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

// Solver Fixo (NÃO salva imagens intermediárias)
template<typename MatrixType>
inline ReconstructionResult run_cgnr_solver_fixed_iter(const Eigen::VectorXd &g_signal, const MatrixType &H_model,
                                                       const int num_iterations) {
    if (H_model.rows() != g_signal.size()) throw std::runtime_error("...");
    if (H_model.cols() <= 0) throw std::runtime_error("...");
    if (H_model.rows() == 0) return ReconstructionResult{};

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
        if (w_norm_sq >= std::numeric_limits<double>::epsilon()) { alpha = z_norm_sq / w_norm_sq; } else {
            /* std::cout << "[AVISO - FixedIter] ||H*p||^2 proximo de zero..." << std::endl; */
            z_norm_sq = 0.0;
        }
        f += alpha * p;
        r -= alpha * w;
        Eigen::VectorXd z_next = H_model.transpose() * r;
        const double z_next_norm_sq = z_next.squaredNorm();
        double beta = 0.0;
        if (z_norm_sq >= std::numeric_limits<double>::epsilon()) { beta = z_next_norm_sq / z_norm_sq; } else {
            /* std::cout << "[AVISO - FixedIter] ||z||^2 proximo de zero..." << std::endl; */
        }
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

// --- Funções FISTA (Novas) ---

template<typename MatrixType>
inline double calculate_lipschitz_constant(const MatrixType &H, int max_power_iters = 20) {
    if (H.cols() == 0) return 1.0;

    Eigen::VectorXd v = Eigen::VectorXd::Random(H.cols());
    v.normalize();

    for (int i = 0; i < max_power_iters; ++i) {
        Eigen::VectorXd H_v = H * v;
        Eigen::VectorXd Ht_H_v = H.transpose() * H_v;
        v = Ht_H_v;
        v.normalize();
    }

    Eigen::VectorXd H_v = H * v;
    double L = H_v.squaredNorm();

    L = L * 1.05; // 5% de margem
    if (L < 1e-6) L = 1.0; // Evita L zero

    std::cout << "[INFO] Constante de Lipschitz (c) estimada: " << L << std::endl;
    return L;
}

inline double soft_threshold(double x, double alpha) {
    if (x > alpha) return x - alpha;
    if (x < -alpha) return x + alpha;
    return 0.0;
}

inline Eigen::VectorXd soft_threshold_vec(const Eigen::VectorXd &x, double alpha) {
    return x.unaryExpr([alpha](double val) { return soft_threshold(val, alpha); });
}

template<typename MatrixType>
inline ReconstructionResult run_fista_solver_save_iters(
    const Eigen::VectorXd &g_signal, const MatrixType &H_model,
    const double tolerance, const int max_iterations,
    const std::string &base_filename_prefix,
    const std::filesystem::path &output_dir,
    int img_rows, int img_cols) {
    if (H_model.rows() != g_signal.size()) throw std::runtime_error(
        "Dimensoes incompativeis: H.rows()!=" + std::to_string(H_model.rows()) + " vs g.size()=" + std::to_string(
            g_signal.size()));
    if (H_model.cols() <= 0) throw std::runtime_error("Matriz H tem " + std::to_string(H_model.cols()) + " colunas.");
    if (H_model.rows() == 0) return ReconstructionResult{};


    const auto start_time = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd f_k = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd y_k = f_k;
    double t_k = 1.0;
    Eigen::VectorXd f_k_next, y_k_next;
    double t_k_next;

    double c = calculate_lipschitz_constant(H_model);
    double step_size = 1.0 / c;

    Eigen::VectorXd Ht_g = H_model.transpose() * g_signal;
    double lambda = 0.0;
    if (Ht_g.size() > 0) {
        lambda = Ht_g.cwiseAbs().maxCoeff() * 0.10;
        constexpr double min_lambda = 1e-9;
        if (lambda < min_lambda) {
            lambda = min_lambda;
            std::cout << "[INFO] Lambda (FISTA) calculado era quase zero, usando piso minimo: " << lambda << std::endl;
        }
    } else {
        lambda = 1e-9;
        std::cout << "[AVISO] Vetor H^T*g (FISTA) inicial vazio, usando lambda=" << lambda << " como fallback." <<
                std::endl;
    }
    std::cout << "[INFO] Lambda (FISTA-L1): " << lambda << std::endl;
    double threshold_param = lambda * step_size;

    ReconstructionResult result;
    result.iterations = 0;
    result.converged = false;
    result.residual_history.clear();
    result.residual_history.reserve(max_iterations);
    result.solution_history.clear();
    result.solution_history.reserve(max_iterations);

    double current_residual_norm = (g_signal - H_model * f_k).norm();
    double previous_residual_norm = current_residual_norm;
    double epsilon = std::numeric_limits<double>::max();

    bool save_iters = !base_filename_prefix.empty() && img_rows > 0 && img_cols > 0;
    if (save_iters) {
        try {
            std::filesystem::path iter_img_path = output_dir / (base_filename_prefix + "_iter_0.csv");
            saveImageVectorToCsv(f_k, iter_img_path.string(), img_rows, img_cols);
        } catch (const std::exception &e) {
            std::cerr << "[AVISO] Falha ao salvar imagem iter 0 (FISTA): " << e.what() << std::endl;
        }
    }

    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1;

        Eigen::VectorXd grad_y = H_model.transpose() * (H_model * y_k - g_signal);
        f_k_next = soft_threshold_vec(y_k - step_size * grad_y, threshold_param);

        t_k_next = (1.0 + std::sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0;
        y_k_next = f_k_next + ((t_k - 1.0) / t_k_next) * (f_k_next - f_k);

        f_k = f_k_next;
        y_k = y_k_next;
        t_k = t_k_next;

        if (save_iters) {
            try {
                std::filesystem::path iter_img_path =
                        output_dir / (base_filename_prefix + "_iter_" + std::to_string(i + 1) + ".csv");
                saveImageVectorToCsv(f_k, iter_img_path.string(), img_rows, img_cols);
            } catch (const std::exception &e) {
                std::cerr << "[AVISO] Falha ao salvar imagem iter " << i + 1 << " (FISTA): " << e.what() << std::endl;
            }
        }

        current_residual_norm = (g_signal - H_model * f_k).norm();
        result.residual_history.push_back(current_residual_norm);
        result.solution_history.push_back(f_k.norm());
        epsilon = std::abs(current_residual_norm - previous_residual_norm);

        if (epsilon < tolerance) {
            result.converged = true;
            std::cout << "[INFO] Convergencia por epsilon (FISTA) atingida na iteracao " << i + 1 << " (epsilon=" <<
                    std::scientific << epsilon << " < " << tolerance << ")" << std::defaultfloat << std::endl;
            break;
        }
        previous_residual_norm = current_residual_norm;

        if (i == max_iterations - 1 && !result.converged) {
            std::cout << "[INFO] Numero maximo de iteracoes (FISTA) (" << max_iterations <<
                    ") atingido sem convergencia por epsilon (ultimo epsilon=" << std::scientific << epsilon <<
                    std::defaultfloat << ")." << std::endl;
        }
    } // Fim do loop

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f_k;
    result.final_error = current_residual_norm;
    result.final_epsilon = epsilon;
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    return result;
}

// **** FIM DO FISTA ****


// --- Instanciações explícitas para os tipos de matrizes ---
template ReconstructionResult run_cgnr_solver_epsilon_save_iters<Eigen::SparseMatrix<double> >(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::filesystem::path &output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_preconditioned_save_iters<Eigen::SparseMatrix<double> >(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::filesystem::path &output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_fixed_iter<Eigen::SparseMatrix<double> >(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    int num_iterations);

// Instanciação do FISTA
template ReconstructionResult run_fista_solver_save_iters<Eigen::SparseMatrix<double> >(
    const Eigen::VectorXd &g_signal, const Eigen::SparseMatrix<double> &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::filesystem::path &output_dir,
    int img_rows, int img_cols);


// Instanciações para Matriz Densa (para reativar testes densos se necessário)
template ReconstructionResult run_cgnr_solver_epsilon_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::filesystem::path &output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_preconditioned_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::filesystem::path &output_dir,
    int img_rows, int img_cols);

template ReconstructionResult run_cgnr_solver_fixed_iter<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    int num_iterations);

template ReconstructionResult run_fista_solver_save_iters<Eigen::MatrixXd>(
    const Eigen::VectorXd &g_signal, const Eigen::MatrixXd &H_model,
    double tolerance, int max_iterations,
    const std::string &base_filename_prefix, const std::filesystem::path &output_dir,
    int img_rows, int img_cols);


// --- Função Principal de Comparação Esparsa ---
// **** CORREÇÃO: TestConfig -> DatasetConfig ****
// Função para encontrar o ponto ótimo na curva L (declarada antes de ser usada)
inline int find_l_curve_corner(const std::vector<double> &residual_norms, const std::vector<double> &solution_norms) {
    if (residual_norms.size() < 3) return residual_norms.size() - 1;

    // Usa log dos valores para melhor análise
    std::vector<double> log_residual(residual_norms.size());
    std::vector<double> log_solution(solution_norms.size());

    // Calcula os logs e normaliza
    double min_res = *std::min_element(residual_norms.begin(), residual_norms.end());
    double max_res = *std::max_element(residual_norms.begin(), residual_norms.end());
    double min_sol = *std::min_element(solution_norms.begin(), solution_norms.end());
    double max_sol = *std::max_element(solution_norms.begin(), solution_norms.end());

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

inline std::pair<PerformanceMetrics, PerformanceMetrics> run_sparse_comparison(const DatasetConfig &config) {
    PerformanceMetrics standard_metrics;
    PerformanceMetrics precond_metrics;
    standard_metrics.optimization_type = "standard";
    precond_metrics.optimization_type = "jacobi";

    // Primeiro roda com iterações suficientes para encontrar o ponto ótimo
    constexpr int initial_iterations = 50; // Número de iterações para construir a curva L
    constexpr int max_iterations = 10;
    constexpr double epsilon_tolerance = 1e-4;

    std::cout << "\n-----------------------------------------------------" << std::endl;
    // **** CORREÇÃO: test_name -> description ****
    std::cout << "Iniciando Comparacao Esparsa para: " << config.description << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    // **** CORREÇÃO: test_name -> name ****
    std::string base_filename = config.name; // Usa o nome curto para arquivos
    // C++17 compatível replace:
    std::replace(base_filename.begin(), base_filename.end(), ' ', '_');
    std::replace(base_filename.begin(), base_filename.end(), '(', '_');
    std::replace(base_filename.begin(), base_filename.end(), ')', '_');
    std::replace(base_filename.begin(), base_filename.end(), '-', '_');


    std::filesystem::path output_dir = "../output_csv";
    // **** CORREÇÃO: h_matrix_path -> h_matrix_csv ****
    std::filesystem::path h_path = config.h_matrix_csv;
    std::filesystem::path data_dir = h_path.parent_path();
    std::filesystem::path sparse_bin_fs_path = data_dir / (h_path.filename().string() + ".sparse.bin");

    // --- Teste 1: CGNR Padrão ---
    try {
        std::cout << "\n--- Rodando CGNR Esparso Padrao (Binario) ---\n";
        auto start_load = std::chrono::high_resolution_clock::now();
        Eigen::SparseMatrix<double> H_std = loadSparseMatrix(sparse_bin_fs_path.string());
        // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
        Eigen::VectorXd g_std = loadVectorData(config.g_signal_csv);
        auto end_load = std::chrono::high_resolution_clock::now();
        standard_metrics.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
        standard_metrics.estimated_ram_mb =
                static_cast<double>(H_std.nonZeros() * (sizeof(double) + sizeof(int)) + (H_std.outerSize() + 1) * sizeof
                                    (int)) / (1024.0 * 1024.0);

        normalize_system_rows(H_std, g_std);
        Eigen::VectorXd z0_std = H_std.transpose() * g_std;
        std::cout << "[DEBUG Standard] Norma de z0 (H^T * g norm): " << z0_std.norm() << std::endl;

        // Define o prefixo do arquivo para o FISTA
        std::string filename_prefix_std = "image_" + base_filename + "_sparse_standard";
        std::string filename_prefix_fista = "image_" + base_filename + "_sparse_fista";
        ReconstructionResult res_std = run_cgnr_solver_epsilon_save_iters(
            g_std, H_std, epsilon_tolerance, max_iterations,
            filename_prefix_std, output_dir, config.image_rows, config.image_cols);

        standard_metrics.solve_time_ms = res_std.execution_time_ms;
        standard_metrics.iterations = res_std.iterations;
        standard_metrics.final_error = res_std.final_error;
        standard_metrics.final_epsilon = res_std.final_epsilon;
        standard_metrics.converged = res_std.converged;

        // Primeiro executa com iterações suficientes para construir a curva L
        ReconstructionResult res_std_initial = run_cgnr_solver_fixed_iter(g_std, H_std, initial_iterations);

        // Encontra o ponto ótimo na curva L
        int optimal_iter = find_l_curve_corner(res_std_initial.residual_history, res_std_initial.solution_history);
        std::filesystem::path hist_path_std =
                output_dir / ("convergence_history_" + base_filename + "_sparse_standard.csv");
        std::filesystem::path lcurve_path_std = output_dir / ("lcurve_" + base_filename + "_sparse_standard.csv");
        saveHistoryToCSV(res_std_initial, hist_path_std.string());
        saveLcurveToCSV(res_std_initial, lcurve_path_std.string());

        // Executa novamente com o número ótimo de iterações
        std::cout << "[INFO] Ponto otimo da curva L encontrado na iteracao " << optimal_iter << std::endl;
        ReconstructionResult res_std_optimal = run_cgnr_solver_fixed_iter(g_std, H_std, optimal_iter);
    } catch (const std::exception &e) {
        std::cerr << "[ERRO - Esparso Padrao] " << e.what() << std::endl;
        standard_metrics = PerformanceMetrics();
        standard_metrics.optimization_type = "standard";
    }

    // --- Teste 2: CGNR Pré-condicionado ---
    try {
        std::cout << "\n--- Rodando CGNR Esparso Pre-condicionado (Binario) ---\n";
        auto start_load = std::chrono::high_resolution_clock::now();
        Eigen::SparseMatrix<double> H_pre = loadSparseMatrix(sparse_bin_fs_path.string());
        // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
        Eigen::VectorXd g_pre = loadVectorData(config.g_signal_csv);
        auto end_load = std::chrono::high_resolution_clock::now();
        precond_metrics.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
        precond_metrics.estimated_ram_mb =
                static_cast<double>(H_pre.nonZeros() * (sizeof(double) + sizeof(int)) + (H_pre.outerSize() + 1) * sizeof
                                    (int)) / (1024.0 * 1024.0);

        normalize_system_rows(H_pre, g_pre);
        Eigen::VectorXd z0_pre = H_pre.transpose() * g_pre;
        std::cout << "[DEBUG Precond] Norma de z0 (H^T * g norm): " << z0_pre.norm() << std::endl;

        std::string filename_prefix_pre = "image_" + base_filename + "_sparse_precond";
        ReconstructionResult res_pre = run_cgnr_solver_preconditioned_save_iters(
            g_pre, H_pre, epsilon_tolerance, max_iterations,
            filename_prefix_pre, output_dir, config.image_rows, config.image_cols);

        precond_metrics.solve_time_ms = res_pre.execution_time_ms;
        precond_metrics.iterations = res_pre.iterations;
        precond_metrics.final_error = res_pre.final_error;
        precond_metrics.final_epsilon = res_pre.final_epsilon;
        precond_metrics.converged = res_pre.converged;

        // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
        // Primeiro executa com iterações suficientes para construir a curva L
        ReconstructionResult res_pre_initial = run_cgnr_solver_fixed_iter(g_pre, H_pre, initial_iterations);

        // Encontra o ponto ótimo na curva L
        int optimal_iter_pre = find_l_curve_corner(res_pre_initial.residual_history, res_pre_initial.solution_history);
        std::cout << "[INFO] Ponto otimo da curva L (precondicionado) encontrado na iteracao " << optimal_iter_pre <<
                std::endl;

        // Executa novamente com o número ótimo de iterações
        ReconstructionResult res_pre_optimal = run_cgnr_solver_fixed_iter(g_pre, H_pre, optimal_iter_pre);
        std::filesystem::path hist_path_pre =
                output_dir / ("convergence_history_" + base_filename + "_sparse_precond.csv");
        // **** CORREÇÃO: Typo "lcurve_"to_string() ****
        std::filesystem::path lcurve_path_pre = output_dir / ("lcurve_" + base_filename + "_sparse_precond.csv");
        saveHistoryToCSV(res_pre_initial, hist_path_pre.string());
        saveLcurveToCSV(res_pre_initial, lcurve_path_pre.string());
    } catch (const std::exception &e) {
        std::cerr << "[ERRO - Esparso Precondicionado] " << e.what() << std::endl;
        precond_metrics = PerformanceMetrics();
        precond_metrics.optimization_type = "jacobi";
    }

    return {standard_metrics, precond_metrics};
}

// Função para executar o FISTA e coletar métricas
inline std::pair<PerformanceMetrics, ReconstructionResult> run_sparse_fista(
    const DatasetConfig &config,
    const std::filesystem::path &output_dir,
    const double tolerance = 1e-6,
    const int max_iterations = 1000) {
    PerformanceMetrics metrics;
    metrics.optimization_type = "fista";

    const auto load_start = std::chrono::high_resolution_clock::now();

    // Carrega a matriz H do arquivo binário esparso
    const std::filesystem::path h_path(config.h_matrix_csv);
    const std::string sparse_bin_path = h_path.parent_path().string() + "/" + h_path.filename().string() +
                                        ".sparse.bin";
    Eigen::SparseMatrix<double> H = loadSparseMatrix(sparse_bin_path);

    // Carrega o sinal g
    Eigen::VectorXd g = loadVectorData(config.g_signal_csv);

    // Normaliza o sistema
    normalize_system_rows(H, g);

    const auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    // Estima uso de RAM (em MB) - apenas para matriz esparsa
    metrics.estimated_ram_mb = (H.nonZeros() * (sizeof(double) + sizeof(int)) +
                                H.outerSize() * sizeof(int)) / (1024.0 * 1024.0);

    // Executa o solver FISTA
    const auto solve_start = std::chrono::high_resolution_clock::now();

    ReconstructionResult result = run_fista_solver_save_iters(
        g, H, tolerance, max_iterations,
        "fista_" + config.name, output_dir,
        config.image_rows, config.image_cols
    );

    const auto solve_end = std::chrono::high_resolution_clock::now();
    metrics.solve_time_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();

    // Atualiza métricas finais
    metrics.iterations = result.iterations;
    metrics.final_error = result.final_error;
    metrics.final_epsilon = result.final_epsilon;
    metrics.converged = result.converged;

    return {metrics, result};
}

// Função para comparar FISTA com os outros métodos
inline std::pair<PerformanceMetrics, std::pair<PerformanceMetrics, PerformanceMetrics> >
run_sparse_comparison_with_fista(const DatasetConfig &config) {
    std::filesystem::path output_dir = "../output_csv";
    constexpr double tolerance = 1e-6;
    constexpr int max_iterations = 1000;
    constexpr int initial_iterations = 50; // Número de iterações para construir a curva L

    // Cria base_filename padronizado
    std::string base_filename = config.name;
    std::replace(base_filename.begin(), base_filename.end(), ' ', '_');
    std::replace(base_filename.begin(), base_filename.end(), '(', '_');
    std::replace(base_filename.begin(), base_filename.end(), ')', '_');
    std::replace(base_filename.begin(), base_filename.end(), '-', '_');

    // Define prefixos padronizados para todos os métodos
    std::string filename_prefix_std = "image_" + base_filename + "_sparse_standard";
    std::string filename_prefix_pre = "image_" + base_filename + "_sparse_precond";
    std::string filename_prefix_fista = "image_" + base_filename + "_sparse_fista";

    // Execute each solver
    PerformanceMetrics standard_metrics;
    standard_metrics.optimization_type = "standard";

    PerformanceMetrics precond_metrics;
    precond_metrics.optimization_type = "precond";

    PerformanceMetrics fista_metrics;
    fista_metrics.optimization_type = "fista";

    try {
        // Standard CGNR
        const auto std_load_start = std::chrono::high_resolution_clock::now();
        std::filesystem::path h_path(config.h_matrix_csv);
        std::string sparse_bin_path = h_path.parent_path().string() + "/" + h_path.filename().string() + ".sparse.bin";
        Eigen::SparseMatrix<double> H_std = loadSparseMatrix(sparse_bin_path);
        Eigen::VectorXd g_std = loadVectorData(config.g_signal_csv);
        const auto std_load_end = std::chrono::high_resolution_clock::now();
        standard_metrics.load_time_ms = std::chrono::duration<double, std::milli>(std_load_end - std_load_start).
                count();

        normalize_system_rows(H_std, g_std);
        auto std_result = run_cgnr_solver_epsilon_save_iters(
            g_std, H_std, tolerance, max_iterations,
            filename_prefix_std, output_dir,
            config.image_rows, config.image_cols
        );

        standard_metrics.solve_time_ms = std_result.execution_time_ms;
        standard_metrics.iterations = std_result.iterations;
        standard_metrics.final_error = std_result.final_error;
        standard_metrics.final_epsilon = std_result.final_epsilon;
        standard_metrics.converged = std_result.converged;

        // Save convergence history
        std::filesystem::path hist_path_std =
                output_dir / ("convergence_history_" + base_filename + "_sparse_standard.csv");
        std::filesystem::path lcurve_path_std = output_dir / ("lcurve_" + base_filename + "_sparse_standard.csv");
        saveHistoryToCSV(std_result.residual_history, hist_path_std.string());
        saveLcurveToCSV(std_result, lcurve_path_std.string());
    } catch (const std::exception &e) {
        std::cerr << "[ERRO] Falha no CGNR padrao: " << e.what() << std::endl;
    }

    try {
        // Preconditioned CGNR
        const auto pre_load_start = std::chrono::high_resolution_clock::now();
        std::filesystem::path h_path(config.h_matrix_csv);
        std::string sparse_bin_path = h_path.parent_path().string() + "/" + h_path.filename().string() + ".sparse.bin";
        Eigen::SparseMatrix<double> H_pre = loadSparseMatrix(sparse_bin_path);
        Eigen::VectorXd g_pre = loadVectorData(config.g_signal_csv);
        const auto pre_load_end = std::chrono::high_resolution_clock::now();
        precond_metrics.load_time_ms = std::chrono::duration<double, std::milli>(pre_load_end - pre_load_start).count();

        normalize_system_rows(H_pre, g_pre);
        auto pre_result = run_cgnr_solver_preconditioned_save_iters(
            g_pre, H_pre, tolerance, max_iterations,
            filename_prefix_pre, output_dir,
            config.image_rows, config.image_cols
        );

        precond_metrics.solve_time_ms = pre_result.execution_time_ms;
        precond_metrics.iterations = pre_result.iterations;
        precond_metrics.final_error = pre_result.final_error;
        precond_metrics.final_epsilon = pre_result.final_epsilon;
        precond_metrics.converged = pre_result.converged;

        // Save convergence history
        std::filesystem::path hist_path_pre =
                output_dir / ("convergence_history_" + base_filename + "_sparse_precond.csv");
        std::filesystem::path lcurve_path_pre = output_dir / ("lcurve_" + base_filename + "_sparse_precond.csv");
        saveHistoryToCSV(pre_result.residual_history, hist_path_pre.string());
        saveLcurveToCSV(pre_result, lcurve_path_pre.string());
    } catch (const std::exception &e) {
        std::cerr << "[ERRO] Falha no CGNR pre-condicionado: " << e.what() << std::endl;
    }

    try {
        // FISTA
        const auto fista_load_start = std::chrono::high_resolution_clock::now();
        std::filesystem::path h_path(config.h_matrix_csv);
        std::string sparse_bin_path = h_path.parent_path().string() + "/" + h_path.filename().string() + ".sparse.bin";
        Eigen::SparseMatrix<double> H_fista = loadSparseMatrix(sparse_bin_path);
        Eigen::VectorXd g_fista = loadVectorData(config.g_signal_csv);
        const auto fista_load_end = std::chrono::high_resolution_clock::now();
        fista_metrics.load_time_ms = std::chrono::duration<double, std::milli>(fista_load_end - fista_load_start).
                count();

        normalize_system_rows(H_fista, g_fista);
        auto fista_result = run_fista_solver_save_iters(
            g_fista, H_fista, tolerance, initial_iterations,
            filename_prefix_fista, output_dir,
            config.image_rows, config.image_cols
        );

        // Encontra o ponto ótimo na curva L
        int optimal_iter_fista = find_l_curve_corner(fista_result.residual_history, fista_result.solution_history);
        std::cout << "[INFO] Ponto otimo da curva L (FISTA) encontrado na iteracao " << optimal_iter_fista << std::endl;

        // Executa novamente com o número ótimo de iterações
        auto fista_result_optimal = run_fista_solver_save_iters(
            g_fista, H_fista, tolerance, optimal_iter_fista,
            filename_prefix_fista, output_dir,
            config.image_rows, config.image_cols
        );

        fista_metrics.solve_time_ms = fista_result.execution_time_ms;
        fista_metrics.iterations = fista_result.iterations;
        fista_metrics.final_error = fista_result.final_error;
        fista_metrics.final_epsilon = fista_result.final_epsilon;
        fista_metrics.converged = fista_result.converged;

        // Save convergence history
        std::filesystem::path hist_path_fista =
                output_dir / ("convergence_history_" + base_filename + "_sparse_fista.csv");
        std::filesystem::path lcurve_path_fista = output_dir / ("lcurve_" + base_filename + "_sparse_fista.csv");
        saveHistoryToCSV(fista_result, hist_path_fista.string());
        saveLcurveToCSV(fista_result, lcurve_path_fista.string());
    } catch (const std::exception &e) {
        std::cerr << "[ERRO] Falha no FISTA: " << e.what() << std::endl;
    }

    return {fista_metrics, {standard_metrics, precond_metrics}};
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
                                 int iteration,
                                 int img_rows, int img_cols) {
    auto images_dir = output_dir / "images";
    std::filesystem::create_directories(images_dir);
    std::filesystem::path iter_img_path =
            images_dir / (base_filename_prefix + "_iter_" + std::to_string(iteration) + ".csv");
    saveImageVectorToCsv(vec, iter_img_path.string(), img_rows, img_cols);
}

inline void save_convergence_data(const ReconstructionResult &result,
                                  const std::filesystem::path &output_dir,
                                  const std::string &base_filename) {
    auto metrics_dir = output_dir / "metrics";
    auto lcurve_dir = output_dir / "lcurve";

    std::filesystem::create_directories(metrics_dir);
    std::filesystem::create_directories(lcurve_dir);

    const std::filesystem::path hist_path = metrics_dir / ("convergence_history_" + base_filename + ".csv");
    const std::filesystem::path lcurve_path = lcurve_dir / ("lcurve_" + base_filename + ".csv");

    saveHistoryToCSV(result, hist_path.string());
    saveLcurveToCSV(result, lcurve_path.string());
}
#endif //ULTRASOUNDSERVER_SOLVER_COMPARISON_HPP
