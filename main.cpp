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
#include <limits> // Para std::numeric_limits
#include <filesystem> // Para manipulação de caminhos (C++17)

#include <Eigen/Dense>
#include <Eigen/Sparse>

// --- Estruturas de Dados ---

struct ReconstructionResult {
    Eigen::VectorXd image; // Vetor f reconstruído
    int iterations{};
    double final_error{}; // Norma do resíduo final ||r||
    double final_epsilon{}; // Valor final de epsilon = | ||r_i+1|| - ||r_i|| |
    double execution_time_ms{};
    bool converged{}; // Indica se parou pela tolerância epsilon
    std::vector<double> residual_history; // Histórico da norma ||r_i||
    std::vector<double> solution_history; // Histórico da norma ||f_i||
};

struct TestConfig {
    std::string test_name;
    std::string h_matrix_path;
    std::string g_signal_path;
    int image_rows;
    int image_cols;
};

struct PerformanceMetrics {
    double load_time_ms = 0.0;
    double solve_time_ms = 0.0;
    double estimated_ram_mb = 0.0;
    int iterations = 0;
    double final_error = 0.0; // ||r|| final
    double final_epsilon = 0.0; // Epsilon final
    bool converged = false;
};

struct TestResult {
    std::string test_name;
    PerformanceMetrics dense_text;
    PerformanceMetrics dense_binary;
    PerformanceMetrics sparse_text;
    PerformanceMetrics sparse_binary;
};


// --- Funções de I/O ---

Eigen::VectorXd loadVectorData(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<double> values;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        // Ignora linhas vazias
        if (line.empty() || std::ranges::all_of(line, ::isspace)) continue;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            try {
                // Remove espaços em branco extras
                std::erase_if(cell, ::isspace);
                if (!cell.empty()) {
                    // Verifica se a célula não está vazia após remover espaços
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
    // Usa Map para evitar cópia desnecessária
    Eigen::Map<Eigen::VectorXd> vec_map(values.data(), values.size());
    // Retorna uma cópia para garantir que o dado persista após 'values' sair de escopo
    return Eigen::VectorXd(vec_map);
}


Eigen::MatrixXd loadDenseData(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<double> values;
    std::string line;
    int rows = 0;
    long long cols = -1; // Inicializa com -1 para detectar a primeira linha
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || std::ranges::all_of(line, ::isspace)) continue; // Skip empty lines

        std::stringstream lineStream(line);
        std::string cell;
        long long current_cols = 0;
        std::vector<double> row_values; // Armazena valores da linha atual

        while (std::getline(lineStream, cell, ',')) {
            try {
                std::erase_if(cell, ::isspace);
                if (!cell.empty()) {
                    row_values.push_back(std::stod(cell));
                } else {
                    // Considera célula vazia como 0.0 ou lança erro? Vamos assumir 0.0 por enquanto.
                    row_values.push_back(0.0);
                    std::cerr << "[AVISO] Celula vazia encontrada em: " << path << ", linha: " << line_num <<
                            ", coluna: " << current_cols + 1 << ". Assumindo 0.0." << std::endl;
                }
                current_cols++;
            } catch (const std::invalid_argument &) {
                std::cerr << "[AVISO] Ignorando valor nao numerico em: " << path << ", linha: " << line_num <<
                        ", celula: '" << cell << "'" << std::endl;
                // Adiciona um valor padrão (e.g., 0) para manter contagem de colunas
                row_values.push_back(0.0);
                current_cols++;
            } catch (const std::out_of_range &) {
                std::cerr << "[AVISO] Ignorando valor fora do range em: " << path << ", linha: " << line_num <<
                        ", celula: '" << cell << "'" << std::endl;
                row_values.push_back(0.0); // Adiciona um valor padrão
                current_cols++;
            }
        }

        if (cols == -1) {
            // Primeira linha válida lida
            if (current_cols == 0) continue; // Pula se a primeira linha válida não tiver colunas
            cols = current_cols;
        } else if (current_cols != cols) {
            std::cerr << "[ERRO] Numero inconsistente de colunas na linha " << line_num << " do arquivo " << path <<
                    ". Esperado: " << cols << ", Encontrado: " << current_cols << ". Abortando." << std::endl;
            throw std::runtime_error("Inconsistencia de colunas no CSV denso.");
        }

        // Adiciona os valores da linha ao vetor principal
        values.insert(values.end(), row_values.begin(), row_values.end());
        rows++;
    }
    if (rows == 0 || cols <= 0 || values.empty()) {
        throw std::runtime_error(
            "Nao foi possivel carregar dados validos da matriz densa de: " + path + " (rows=" + std::to_string(rows) +
            ", cols=" + std::to_string(cols) + ")");
    }

    // Cria a matriz Eigen (RowMajor por causa do Map)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > mat_map(
        values.data(), rows, cols);
    // Retorna uma cópia para garantir persistência
    return Eigen::MatrixXd(mat_map);
}


Eigen::SparseMatrix<double> convertCsvToSparse(const std::string &path, int expected_cols) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
    std::vector<Eigen::Triplet<double> > tripletList;
    std::string line;
    int row = 0;
    long long actual_cols = -1;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || std::ranges::all_of(line, ::isspace)) continue; // Skip empty lines

        std::stringstream lineStream(line);
        std::string cell;
        int col = 0;
        while (std::getline(lineStream, cell, ',')) {
            try {
                std::erase_if(cell, ::isspace);
                if (!cell.empty()) {
                    double value = std::stod(cell);
                    if (std::abs(value) > 1e-12) {
                        // Only store non-zeros
                        tripletList.emplace_back(row, col, value);
                    }
                }
                // Não adicionamos nada para células vazias em esparsas
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
            // Primeira linha válida
            if (col == 0) continue; // Ignora se a primeira linha válida não teve colunas
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

    Eigen::SparseMatrix<double> mat(row, actual_cols); // Usa linhas e colunas reais lidas
    if (!tripletList.empty()) {
        mat.setFromTriplets(tripletList.begin(), tripletList.end());
    } else {
        std::cerr << "[AVISO] Nenhum elemento nao-zero (acima de 1e-12) encontrado em " << path << std::endl;
    }
    mat.makeCompressed(); // Garante que está no formato comprimido
    return mat;
}


void saveSparseMatrix(const Eigen::SparseMatrix<double> &mat, const std::string &path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel criar o arquivo binario: " + path);

    // Garante que a matriz está comprimida antes de salvar
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

    // Verifica se os ponteiros são válidos antes de escrever
    if (nonZeros > 0) {
        if (!compressed_mat.valuePtr() || !compressed_mat.innerIndexPtr()) {
            throw std::runtime_error("Ponteiros internos invalidos ao salvar matriz esparsa para: " + path);
        }
        file.write(reinterpret_cast<const char *>(compressed_mat.valuePtr()), nonZeros * sizeof(double));
        file.write(reinterpret_cast<const char *>(compressed_mat.innerIndexPtr()), nonZeros * sizeof(int));
        // Eigen usa int
    }
    if (!compressed_mat.outerIndexPtr()) {
        throw std::runtime_error("Ponteiro outerIndex invalido ao salvar matriz esparsa para: " + path);
    }
    file.write(reinterpret_cast<const char *>(compressed_mat.outerIndexPtr()), (outerSize + 1) * sizeof(int));
    // Eigen usa int

    if (!file) throw std::runtime_error("Erro ao escrever no arquivo binario esparso: " + path);
    file.close();
    if (!file) throw std::runtime_error("Erro ao fechar o arquivo binario esparso: " + path);
}


Eigen::SparseMatrix<double> loadSparseMatrix(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " + path);
    Eigen::Index rows, cols;
    Eigen::Index nonZeros; // Use Index para nonZeros também
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char *>(&nonZeros), sizeof(nonZeros));

    if (!file || file.eof()) throw std::runtime_error("Erro ao ler cabecalho do arquivo binario esparso: " + path);
    if (rows < 0 || cols < 0 || nonZeros < 0) throw std::runtime_error(
        "Dimensoes invalidas (" + std::to_string(rows) + "x" + std::to_string(cols) + ", nnz=" +
        std::to_string(nonZeros) + ") lidas do arquivo binario esparso: " + path);


    Eigen::SparseMatrix<double> mat(rows, cols);
    mat.makeCompressed(); // Importante antes de acessar ponteiros
    mat.resizeNonZeros(nonZeros); // Aloca espaço

    // **CORREÇÃO: Removido check mat.capacity()**
    // A verificação de sucesso da alocação é implícita; se falhar,
    // a leitura subsequente provavelmente causará um crash ou erro.
    // Eigen não garante std::bad_alloc em todos os casos.

    // Leitura direta nos buffers alocados
    if (nonZeros > 0) {
        if (!mat.valuePtr() || !mat.innerIndexPtr()) {
            // Se resizeNonZeros falhou silenciosamente, os ponteiros podem ser nulos
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
        // Verifica o estado do stream APÓS as leituras
        if (file.eof()) {
            std::cerr << "[AVISO] Fim de arquivo prematuro ao ler dados de " << path <<
                    ". A matriz pode estar incompleta." << std::endl;
            // Decide se lança erro ou continua com dados parciais. Lançar é mais seguro.
            throw std::runtime_error("Fim de arquivo inesperado ao ler dados do arquivo binario esparso: " + path);
        } else {
            throw std::runtime_error("Erro de leitura ao processar dados do arquivo binario esparso: " + path);
        }
    }


    mat.finalize(); // Pode ajudar a validar a estrutura interna
    return mat;
}


void saveDenseMatrix(const Eigen::MatrixXd &mat, const std::string &path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("Nao foi possivel criar o arquivo binario: " + path);
    const auto rows = mat.rows(), cols = mat.cols();
    file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char *>(mat.data()),
               static_cast<std::streamsize>(rows) * cols * sizeof(double)); // Cast
    if (!file) throw std::runtime_error("Erro ao escrever no arquivo binario denso: " + path);
    file.close();
    if (!file) throw std::runtime_error("Erro ao fechar o arquivo binario denso: " + path);
}

Eigen::MatrixXd loadDenseMatrix(const std::string &path) {
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
    // Cast para streamsize
    if (!file) {
        if (file.eof() && (static_cast<std::streamsize>(rows) * cols * sizeof(double) > 0)) {
            // Check if EOF happened unexpectedly
            throw std::runtime_error("Fim de arquivo inesperado ao ler dados do arquivo binario denso: " + path);
        } else if (!file.eof()) {
            // Check for other read errors
            throw std::runtime_error("Erro de leitura ao processar dados do arquivo binario denso: " + path);
        }
        // Se chegou em EOF e leu 0 bytes (matriz 0xN ou Nx0), está ok.
    }

    return mat;
}

void saveHistoryToCSV(const std::vector<double> &history, const std::string &filename) {
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
    } else {
        std::cout << "[INFO] Historico de convergencia salvo em: " << filename << std::endl;
    }
    file.close();
}


void saveLcurveToCSV(const ReconstructionResult &result, const std::string &filename) {
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


void saveImageVectorToCsv(const Eigen::VectorXd &vec, const std::string &filename, int img_rows, int img_cols) {
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

    file << std::scientific << std::setprecision(8); // Define formato

    // Salva linha por linha no CSV
    for (int i = 0; i < img_rows; ++i) {
        for (int j = 0; j < img_cols; ++j) {
            // Calcula o índice linear correto (assumindo ColMajor default do Eigen::VectorXd)
            long long index = static_cast<long long>(j) * img_rows + i;
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


// --- Algoritmos de Reconstrução ---

#include <Eigen/Core> // Para maxCoeff()

/**
 * @brief Executa o solver CGNR com Regularização de Tikhonov.
 * Para até que epsilon = | ||r_i+1|| - ||r_i|| | < tolerance OU max_iterations seja atingido.
 * Assume que H e g JÁ FORAM NORMALIZADOS por linha antes de chamar esta função.
 * @tparam MatrixType Eigen::MatrixXd ou Eigen::SparseMatrix<double>
 * @param g_signal Sinal medido (vetor g) JÁ NORMALIZADO.
 * @param H_model Matriz do sistema (H) JÁ NORMALIZADA.
 * @param tolerance Tolerância para o critério de parada epsilon.
 * @param max_iterations Número máximo de iterações.
 * @return ReconstructionResult Contendo a imagem, métricas e histórico.
 */
template<typename MatrixType>
ReconstructionResult run_cgnr_solver_epsilon(const Eigen::VectorXd &g_signal, const MatrixType &H_model,
                                             const double tolerance = 1e-4, const int max_iterations = 10) {
    // Verificações iniciais
     if (H_model.rows() != g_signal.size()) {
        throw std::runtime_error("Dimensoes incompativeis: H.rows()=" + std::to_string(H_model.rows()) + " != g.size()=" + std::to_string(g_signal.size()));
    }
    if (H_model.cols() <= 0) {
        throw std::runtime_error("Matriz H tem " + std::to_string(H_model.cols()) + " colunas.");
    }
    if (H_model.rows() == 0) {
         std::cerr << "[AVISO] Matriz H ou sinal g estao vazios." << std::endl;
         return ReconstructionResult{}; // Retorna resultado vazio
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    // --- Inicialização ---
    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols()); // f_0 = 0
    Eigen::VectorXd r = g_signal; // r_0 = g (pois f_0 = 0 e g já está normalizado)
    Eigen::VectorXd z = H_model.transpose() * r; // z_0 = H^T * r_0 (usando H normalizada)
    Eigen::VectorXd p = z; // p_0 = z_0
    double z_norm_sq = z.squaredNorm(); // ||z_0||^2

    // --- Cálculo do Lambda (usando o z inicial dos dados normalizados) ---
    double lambda = 0.0;
    if (z.size() > 0) {
        lambda = z.cwiseAbs().maxCoeff() * 0.10; // Fórmula do professor
        if (constexpr double min_lambda = 1e-9; lambda < min_lambda) {
            lambda = min_lambda;
            std::cout << "[INFO] Lambda calculado (" << z.cwiseAbs().maxCoeff() * 0.10
                      << ") era muito pequeno, usando piso minimo: " << lambda << std::endl;
        }
    } else {
        lambda = 1e-9; // Fallback se z for vazio
         std::cout << "[AVISO] Vetor z inicial vazio, usando lambda=" << lambda << " como fallback." << std::endl;
    }
    std::cout << "[INFO] Parametro de regularizacao Lambda (calculado pos-norm): " << lambda << std::endl;
    // --- Fim do cálculo de Lambda ---


    // Inicialização da estrutura de resultados
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


    // --- Loop CGNR Modificado com Tikhonov ---
    for (int i = 0; i < max_iterations; ++i) {
        result.iterations = i + 1; // Contador de iteração (1 a max_iterations)

        Eigen::VectorXd w = H_model * p; // w_i = H * p_i

        // Denominador modificado: ||H * p_i||^2 + lambda * ||p_i||^2
        double p_norm_sq = p.squaredNorm();
        double modified_denominator = w.squaredNorm() + lambda * p_norm_sq;

        // Verificação para evitar divisão por zero
        if (modified_denominator < std::numeric_limits<double>::epsilon()) {
             std::cout << "[INFO] Denominador modificado (||Hp||^2 + lambda*||p||^2 = " << modified_denominator
                       << ") proximo de zero na iteracao " << i + 1 << ". Parando." << std::endl;
             break; // Para o loop
        }

        // Calcula alpha
        double alpha = z_norm_sq / modified_denominator;

        // Atualiza a solução e o resíduo
        f += alpha * p; // f_{i+1} = f_i + alpha_i * p_i
        r -= alpha * w; // r_{i+1} = r_i - alpha_i * w_i

        // Calcula o 'z' modificado para Tikhonov
        Eigen::VectorXd z_next = (H_model.transpose() * r) - (lambda * f); // z_{i+1} = H^T * r_{i+1} - lambda * f_{i+1}
        const double z_next_norm_sq = z_next.squaredNorm();

        // Calcula normas atuais para histórico e critério de parada
        current_residual_norm = r.norm(); // Norma do resíduo ||g - Hf||
        result.residual_history.push_back(current_residual_norm);
        result.solution_history.push_back(f.norm()); // Norma da solução ||f||

        // Calcula epsilon (diferença das normas do resíduo)
        epsilon = std::abs(current_residual_norm - previous_residual_norm);

        // Verifica critério de parada epsilon
        if (epsilon < tolerance) {
            std::cout << "[INFO] Convergencia por epsilon atingida na iteracao " << i+1 << " (epsilon=" << std::scientific << epsilon << " < " << tolerance << ")" << std::defaultfloat << std::endl;
            result.converged = true; // Marca como convergido por epsilon
            break; // Para o loop
        }
        previous_residual_norm = current_residual_norm; // Atualiza norma anterior para próxima iteração

        // Calcula beta
        double beta = 0.0; // Valor padrão se z_norm_sq for zero
        if (z_norm_sq >= std::numeric_limits<double>::epsilon()) {
            beta = z_next_norm_sq / z_norm_sq; // beta_i = ||z_{i+1}||^2 / ||z_i||^2
        } else {
             std::cout << "[INFO] ||z||^2 (" << z_norm_sq << ") proximo de zero na iteracao " << i + 1 << ". Usando beta=0 (restart)." << std::endl;
             // Se z_next também for zero, o algoritmo estagnou
             if (z_next_norm_sq < std::numeric_limits<double>::epsilon()){
                  std::cout << "[INFO] ||z_next||^2 tambem proximo de zero. Provavelmente estagnou. Parando." << std::endl;
                  break; // Para o loop
             }
        }

        // Atualiza a direção de busca 'p' e 'z'
        p = z_next + beta * p; // p_{i+1} = z_{i+1} + beta_i * p_i
        z = z_next;
        z_norm_sq = z_next_norm_sq;

        // Mensagem se atingiu o máximo de iterações sem convergir por epsilon
        if (i == max_iterations - 1 && !result.converged) {
           std::cout << "[INFO] Numero maximo de iteracoes (" << max_iterations << ") atingido sem convergencia por epsilon (ultimo epsilon=" << std::scientific << epsilon << std::defaultfloat << ")." << std::endl;
        }
    } // Fim do loop for

    const auto end_time = std::chrono::high_resolution_clock::now();

    // Preenche os resultados finais
    result.image = f;
    result.final_error = current_residual_norm; // Última norma do resíduo calculada
    result.final_epsilon = epsilon; // Último epsilon calculado
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result; // Retorna a estrutura com todos os resultados
}

template<typename MatrixType>
ReconstructionResult run_cgnr_solver_fixed_iter(const Eigen::VectorXd &g_signal, const MatrixType &H_model,
                                                const int num_iterations = 10) {
    // ... (código mantido como na versão anterior, com a correção do 'beta' já feita) ...
    if (H_model.rows() != g_signal.size()) {
        throw std::runtime_error(
            "Dimensoes incompativeis: H.rows()=" + std::to_string(H_model.rows()) + " != g.size()=" + std::to_string(
                g_signal.size()));
    }
    if (H_model.cols() <= 0) {
        throw std::runtime_error("Matriz H tem " + std::to_string(H_model.cols()) + " colunas.");
    }
    if (H_model.rows() == 0) {
        std::cerr << "[AVISO - FixedIter] Matriz H ou sinal g estao vazios." << std::endl;
        return ReconstructionResult{};
    }


    const auto start_time = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd f = Eigen::VectorXd::Zero(H_model.cols());
    Eigen::VectorXd r = g_signal; // r_0 = g (assuming f_0 = 0)
    Eigen::VectorXd z = H_model.transpose() * r;
    Eigen::VectorXd p = z;
    double z_norm_sq = z.squaredNorm();

    ReconstructionResult result;
    result.iterations = 0;
    result.converged = false; // Not relevant for fixed iter
    result.residual_history.clear();
    result.residual_history.reserve(num_iterations);
    result.solution_history.clear();
    result.solution_history.reserve(num_iterations);

    for (int i = 0; i < num_iterations; ++i) {
        result.iterations = i + 1; // Iteration count starts from 1

        // Store norms *before* update for iteration 'i+1' history
        double current_residual_norm = r.norm();
        result.residual_history.push_back(current_residual_norm);
        double current_solution_norm = f.norm();
        result.solution_history.push_back(current_solution_norm);

        Eigen::VectorXd w = H_model * p;
        double w_norm_sq = w.squaredNorm();

        double alpha = 0.0; // Default alpha if w is zero
        if (w_norm_sq >= std::numeric_limits<double>::epsilon()) {
            alpha = z_norm_sq / w_norm_sq;
        } else {
            std::cout << "[AVISO - FixedIter] ||H*p||^2 proximo de zero na iteracao " << i + 1 << ". Usando alpha=0." <<
                    std::endl;
            z_norm_sq = 0.0; // Ensures beta will be 0 next
        }

        f += alpha * p;
        r -= alpha * w; // If alpha is 0, r remains unchanged

        Eigen::VectorXd z_next = H_model.transpose() * r;
        const double z_next_norm_sq = z_next.squaredNorm();

        double beta = 0.0; // Default beta if division by zero occurs
        if (z_norm_sq >= std::numeric_limits<double>::epsilon()) {
            beta = z_next_norm_sq / z_norm_sq;
        } else {
            std::cout << "[AVISO - FixedIter] ||z||^2 proximo de zero na iteracao " << i + 1 << ". Usando beta=0." <<
                    std::endl;
        }

        p = z_next + beta * p;
        z = z_next;
        z_norm_sq = z_next_norm_sq;
    } // End for loop

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.image = f; // Imagem final após N iterações
    result.final_error = r.norm(); // Erro final após N iterações
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}

// --- Função para Normalizar Linhas ---
template<typename MatrixType>
void normalize_system_rows(MatrixType& H, Eigen::VectorXd& g) {
    if (H.rows() != g.size()) {
        throw std::runtime_error("normalize_system_rows: Dimensoes H/g incompativeis.");
    }
    if (H.rows() == 0) return; // Nada a fazer

    std::cout << "[INFO] Normalizando linhas de H e elementos de g..." << std::endl;
    constexpr double epsilon_norm = 1e-12; // Limiar para evitar divisao por zero

    if constexpr (std::is_same_v<MatrixType, Eigen::MatrixXd>) {
        // --- Versão Densa ---
        #pragma omp parallel for // Paraleliza a normalização das linhas
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
        // Iterar sobre linhas em matriz esparsa é menos direto.
        // É mais eficiente iterar sobre os não-zeros e calcular normas por linha.
        std::vector<double> row_norms_sq(H.rows(), 0.0);

        // Calcula norma ao quadrado de cada linha (mais eficiente para esparsa)
        for (int k=0; k<H.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(H,k); it; ++it) {
                 // it.row() é o índice da linha
                 // it.value() é o valor
                 row_norms_sq[it.row()] += it.value() * it.value();
            }
        }

        // Agora aplica a normalização (modificando os valores não-zeros)
        // Isso precisa ser feito com cuidado para modificar a matriz esparsa
        // Criamos uma nova lista de triplets normalizada
         std::vector<Eigen::Triplet<double>> triplets_normalized;
         triplets_normalized.reserve(H.nonZeros());

        for (int k=0; k<H.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(H,k); it; ++it) {
                double row_norm = std::sqrt(row_norms_sq[it.row()]);
                 if (row_norm > epsilon_norm) {
                     triplets_normalized.emplace_back(it.row(), it.col(), it.value() / row_norm);
                 }
                 // Se row_norm <= epsilon_norm, o elemento efetivamente se torna zero e não é adicionado
            }
        }
         // Atualiza a matriz H com os triplets normalizados
         H.setFromTriplets(triplets_normalized.begin(), triplets_normalized.end());
         // H.makeCompressed(); // setFromTriplets já deve comprimir

         // Normaliza g (isso pode ser paralelizado se g for grande)
         #pragma omp parallel for
         for(Eigen::Index i = 0; i < g.size(); ++i) {
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

// --- Função Principal ---
// --- Função Principal ---
int main(int argc, char *argv[]) {
    std::cout << "======================================================" << std::endl;
    std::cout << " Comparativo de Desempenho: Denso vs. Esparso" << std::endl;
    std::cout << "======================================================" << std::endl;

    // **** DECLARAÇÕES CORRIGIDAS ****
    bool run_dense = true;
    bool run_sparse = true;

    std::vector<TestConfig> tests = {
        {"60x60 (Sinal G-1)", "../data/H-1.csv", "../data/G-1.csv", 60, 60},
        {"60x60 (Sinal G-2)", "../data/H-1.csv", "../data/G-2.csv", 60, 60},
        {"30x30 (Sinal g-1)", "../data/H-2.csv", "../data/g-30x30-1.csv", 30, 30},
        {"30x30 (Sinal g-2)", "../data/H-2.csv", "../data/g-30x30-2.csv", 30, 30}
    };
     // **** FIM DAS DECLARAÇÕES CORRIGIDAS ****


    Eigen::setNbThreads(omp_get_max_threads());
    std::cout << "\n[INFO] Usando " << Eigen::nbThreads() << " threads para os calculos Eigen.\n" << std::endl;

     // Tolerâncias e limites

    std::cout << "[INFO] Verificando/Criando arquivos binarios para acelerar a leitura..." << std::endl;
    for (const auto &config: tests) { // Usa a variável 'tests' declarada acima
        int image_size = config.image_rows * config.image_cols;
        std::filesystem::path h_path = config.h_matrix_path;
        if (!std::filesystem::exists(h_path)) {
            std::cerr << "[ERRO] Arquivo CSV da matriz H nao encontrado: " << config.h_matrix_path << std::endl;
            return 1;
        }
        std::filesystem::path data_dir = h_path.parent_path();

        if (run_sparse) { // Usa a variável 'run_sparse' declarada acima
            std::filesystem::path sparse_bin_fs_path = data_dir / (h_path.filename().string() + ".sparse.bin");
            std::string sparse_bin_path = sparse_bin_fs_path.string();
            if (!std::filesystem::exists(sparse_bin_fs_path)) {
                std::cout << "[AVISO] Criando arquivo binario esparso para " << config.h_matrix_path << " em " <<
                        sparse_bin_path << "..." << std::endl;
                try {
                    saveSparseMatrix(convertCsvToSparse(config.h_matrix_path, image_size), sparse_bin_path);
                    std::cout << "[SUCESSO] Arquivo binario esparso criado: " << sparse_bin_path << std::endl;
                } catch (const std::exception &e) {
                    std::cerr << "[ERRO] Falha ao criar arquivo binario esparso para " << config.h_matrix_path << ": "
                            << e.what() << std::endl;
                    return 1;
                }
            }
        }
        if (run_dense) { // Usa a variável 'run_dense' declarada acima
            std::filesystem::path dense_bin_fs_path = data_dir / (h_path.filename().string() + ".dense.bin");
            std::string dense_bin_path = dense_bin_fs_path.string();
            if (!std::filesystem::exists(dense_bin_fs_path)) {
                std::cout << "[AVISO] Criando arquivo binario denso para " << config.h_matrix_path << " em " <<
                        dense_bin_path << "..." << std::endl;
                try {
                    saveDenseMatrix(loadDenseData(config.h_matrix_path), dense_bin_path);
                    std::cout << "[SUCESSO] Arquivo binario denso criado: " << dense_bin_path << std::endl;
                } catch (const std::exception &e) {
                    std::cerr << "[ERRO] Falha ao criar arquivo binario denso para " << config.h_matrix_path << ": " <<
                            e.what() << std::endl;
                    return 1;
                }
            }
        }
    }
    std::cout << "[INFO] Pre-processamento concluido.\n" << std::endl;


    std::vector<TestResult> all_results;

    // Usa structured binding (C++17) for clarity, referenciando 'tests'
    for (const auto &[test_name, h_matrix_path, g_signal_path, image_rows, image_cols]: tests) {
        constexpr int fixed_iterations_for_csv = 10;
        constexpr int max_iterations = 10;
        constexpr double epsilon_tolerance = 1e-4;
        TestResult current_test;
        current_test.test_name = test_name;
        int image_size = image_rows * image_cols;
        std::cout << "\n========================================\nINICIANDO TESTE: " << test_name <<
                "\n========================================" << std::endl;

        ReconstructionResult res_epsilon;
        ReconstructionResult res_fixed;

        std::string base_filename = test_name;
        std::ranges::replace(base_filename, ' ', '_');
        std::ranges::replace(base_filename, '(', '_');
        std::ranges::replace(base_filename, ')', '_');
        std::ranges::replace(base_filename, '-', '_');

        std::filesystem::path output_dir = "../output_csv";
        try {
            std::filesystem::create_directories(output_dir);
        } catch (const std::filesystem::filesystem_error &e) {
            std::cerr << "[ERRO] Nao foi possivel criar o diretorio de saida: " << output_dir << " - " << e.what() <<
                    std::endl;
            return 1;
        }


        if (run_dense) { // Usa a variável 'run_dense' declarada acima
            try {
                std::cout << "\n--- 1. Denso de Texto ---\n";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::MatrixXd H = loadDenseData(h_matrix_path);
                Eigen::VectorXd g = loadVectorData(g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.dense_text.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).
                        count();
                current_test.dense_text.estimated_ram_mb =
                        static_cast<double>(H.rows()) * H.cols() * sizeof(double) / (1024.0 * 1024.0);

                // Normalização
                normalize_system_rows(H, g);

                // Debug z0 pós-normalização
                Eigen::VectorXd z0_dense_text = H.transpose() * g;
                std::cout << "[DEBUG Dense Txt Norm] Norma de z0 (H^T * g): " << z0_dense_text.norm() << std::endl;

                res_epsilon = run_cgnr_solver_epsilon(g, H, epsilon_tolerance, max_iterations);
                 current_test.dense_text.solve_time_ms = res_epsilon.execution_time_ms;
                 current_test.dense_text.iterations = res_epsilon.iterations;
                 current_test.dense_text.final_error = res_epsilon.final_error;
                 current_test.dense_text.final_epsilon = res_epsilon.final_epsilon;
                 current_test.dense_text.converged = res_epsilon.converged;

                std::filesystem::path img_csv_path = output_dir / ("image_" + base_filename + "_dense_text.csv");
                saveImageVectorToCsv(res_epsilon.image, img_csv_path.string(), image_rows, image_cols);

                 // Recarrega e normaliza para fixed_iter
                 Eigen::MatrixXd H_fixed = loadDenseData(h_matrix_path);
                 Eigen::VectorXd g_fixed = loadVectorData(g_signal_path);
                 normalize_system_rows(H_fixed, g_fixed);
                 res_fixed = run_cgnr_solver_fixed_iter(g_fixed, H_fixed, fixed_iterations_for_csv);
                 std::filesystem::path hist_path = output_dir / ("convergence_history_" + base_filename + "_dense_text.csv");
                 std::filesystem::path lcurve_path = output_dir / ("lcurve_" + base_filename + "_dense_text.csv");
                 saveHistoryToCSV(res_fixed.residual_history, hist_path.string());
                 saveLcurveToCSV(res_fixed, lcurve_path.string());

            } catch (const std::exception &e) { std::cerr << "[ERRO - Denso Texto] " << e.what() << std::endl; }

            try {
                std::cout << "\n--- 2. Denso de Binario ---\n";
                 std::filesystem::path h_path = h_matrix_path;
                 std::filesystem::path data_dir = h_path.parent_path();
                 std::filesystem::path dense_bin_fs_path = data_dir / (h_path.filename().string() + ".dense.bin");
                 auto start_load = std::chrono::high_resolution_clock::now();
                 Eigen::MatrixXd H = loadDenseMatrix(dense_bin_fs_path.string());
                 Eigen::VectorXd g = loadVectorData(g_signal_path);
                 auto end_load = std::chrono::high_resolution_clock::now();
                 current_test.dense_binary.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                 current_test.dense_binary.estimated_ram_mb = static_cast<double>(H.rows()) * H.cols() * sizeof(double) / (1024.0 * 1024.0);

                 // Normalização
                 normalize_system_rows(H, g);

                 // Debug z0 pós-normalização
                 Eigen::VectorXd z0_dense_bin = H.transpose() * g;
                 std::cout << "[DEBUG Dense Bin Norm] Norma de z0 (H^T * g): " << z0_dense_bin.norm() << std::endl;

                res_epsilon = run_cgnr_solver_epsilon(g, H, epsilon_tolerance, max_iterations);
                current_test.dense_binary.solve_time_ms = res_epsilon.execution_time_ms;
                current_test.dense_binary.iterations = res_epsilon.iterations;
                current_test.dense_binary.final_error = res_epsilon.final_error;
                current_test.dense_binary.final_epsilon = res_epsilon.final_epsilon;
                current_test.dense_binary.converged = res_epsilon.converged;

                std::filesystem::path img_csv_path = output_dir / ("image_" + base_filename + "_dense_bin.csv");
                saveImageVectorToCsv(res_epsilon.image, img_csv_path.string(), image_rows, image_cols);

                 Eigen::MatrixXd H_fixed = loadDenseMatrix(dense_bin_fs_path.string());
                 Eigen::VectorXd g_fixed = loadVectorData(g_signal_path);
                 normalize_system_rows(H_fixed, g_fixed);
                 res_fixed = run_cgnr_solver_fixed_iter(g_fixed, H_fixed, fixed_iterations_for_csv);
                 std::filesystem::path hist_path = output_dir / ("convergence_history_" + base_filename + "_dense_bin.csv");
                 std::filesystem::path lcurve_path = output_dir / ("lcurve_" + base_filename + "_dense_bin.csv");
                saveHistoryToCSV(res_fixed.residual_history, hist_path.string());
                saveLcurveToCSV(res_fixed, lcurve_path.string());


            } catch (const std::exception &e) { std::cerr << "[ERRO - Denso Binario] " << e.what() << std::endl; }
        }

        if (run_sparse) { // Usa a variável 'run_sparse' declarada acima
            try {
                std::cout << "\n--- 3. Esparso de Texto (Conversao) ---\n";
                auto start_load = std::chrono::high_resolution_clock::now();
                Eigen::SparseMatrix<double> H = convertCsvToSparse(h_matrix_path, image_size);
                Eigen::VectorXd g = loadVectorData(g_signal_path);
                auto end_load = std::chrono::high_resolution_clock::now();
                current_test.sparse_text.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                current_test.sparse_text.estimated_ram_mb = static_cast<double>(H.nonZeros() * (sizeof(double) + sizeof(int)) + (H.outerSize() + 1) * sizeof(int)) / (1024.0 * 1024.0);

                 // Normalização
                 normalize_system_rows(H, g);

                 // Debug z0 pós-normalização
                 Eigen::VectorXd z0_sparse_text = H.transpose() * g;
                 std::cout << "[DEBUG Sparse Txt Norm] Norma de z0 (H^T * g): " << z0_sparse_text.norm() << std::endl;

                res_epsilon = run_cgnr_solver_epsilon(g, H, epsilon_tolerance, max_iterations);
                 current_test.sparse_text.solve_time_ms = res_epsilon.execution_time_ms;
                 current_test.sparse_text.iterations = res_epsilon.iterations;
                 current_test.sparse_text.final_error = res_epsilon.final_error;
                 current_test.sparse_text.final_epsilon = res_epsilon.final_epsilon;
                 current_test.sparse_text.converged = res_epsilon.converged;

                std::filesystem::path img_csv_path = output_dir / ("image_" + base_filename + "_sparse_text.csv");
                saveImageVectorToCsv(res_epsilon.image, img_csv_path.string(), image_rows, image_cols);

                 // Recarrega e normaliza para fixed_iter
                 Eigen::SparseMatrix<double> H_fixed = convertCsvToSparse(h_matrix_path, image_size);
                 Eigen::VectorXd g_fixed = loadVectorData(g_signal_path);
                 normalize_system_rows(H_fixed, g_fixed);
                 res_fixed = run_cgnr_solver_fixed_iter(g_fixed, H_fixed, fixed_iterations_for_csv);
                 std::filesystem::path hist_path = output_dir / ("convergence_history_" + base_filename + "_sparse_text.csv");
                 std::filesystem::path lcurve_path = output_dir / ("lcurve_" + base_filename + "_sparse_text.csv");
                saveHistoryToCSV(res_fixed.residual_history, hist_path.string());
                saveLcurveToCSV(res_fixed, lcurve_path.string());


            } catch (const std::exception &e) { std::cerr << "[ERRO - Esparso Texto] " << e.what() << std::endl; }

            try {
                std::cout << "\n--- 4. Esparso de Binario ---\n";
                 std::filesystem::path h_path = h_matrix_path;
                 std::filesystem::path data_dir = h_path.parent_path();
                 std::filesystem::path sparse_bin_fs_path = data_dir / (h_path.filename().string() + ".sparse.bin");
                 auto start_load = std::chrono::high_resolution_clock::now();
                 Eigen::SparseMatrix<double> H = loadSparseMatrix(sparse_bin_fs_path.string());
                 Eigen::VectorXd g = loadVectorData(g_signal_path);
                 auto end_load = std::chrono::high_resolution_clock::now();
                 current_test.sparse_binary.load_time_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
                 current_test.sparse_binary.estimated_ram_mb = static_cast<double>(H.nonZeros() * (sizeof(double) + sizeof(int)) + (H.outerSize() + 1) * sizeof(int)) / (1024.0 * 1024.0);

                 // Normalização
                 normalize_system_rows(H, g);

                 // Debug z0 pós-normalização
                 Eigen::VectorXd z0_sparse_bin = H.transpose() * g;
                 std::cout << "[DEBUG Sparse Bin Norm] Norma de z0 (H^T * g): " << z0_sparse_bin.norm() << std::endl;

                res_epsilon = run_cgnr_solver_epsilon(g, H, epsilon_tolerance, max_iterations);
                current_test.sparse_binary.solve_time_ms = res_epsilon.execution_time_ms;
                current_test.sparse_binary.iterations = res_epsilon.iterations;
                current_test.sparse_binary.final_error = res_epsilon.final_error;
                current_test.sparse_binary.final_epsilon = res_epsilon.final_epsilon;
                current_test.sparse_binary.converged = res_epsilon.converged;

                std::filesystem::path img_csv_path = output_dir / ("image_" + base_filename + "_sparse_bin.csv");
                saveImageVectorToCsv(res_epsilon.image, img_csv_path.string(), image_rows, image_cols);

                // Recarrega e normaliza para fixed_iter
                 Eigen::SparseMatrix<double> H_fixed = loadSparseMatrix(sparse_bin_fs_path.string());
                 Eigen::VectorXd g_fixed = loadVectorData(g_signal_path);
                 normalize_system_rows(H_fixed, g_fixed);
                 res_fixed = run_cgnr_solver_fixed_iter(g_fixed, H_fixed, fixed_iterations_for_csv);
                 std::filesystem::path hist_path = output_dir / ("convergence_history_" + base_filename + "_sparse_bin.csv");
                 std::filesystem::path lcurve_path = output_dir / ("lcurve_" + base_filename + "_sparse_bin.csv");
                saveHistoryToCSV(res_fixed.residual_history, hist_path.string());
                saveLcurveToCSV(res_fixed, lcurve_path.string());


            } catch (const std::exception &e) { std::cerr << "[ERRO - Esparso Binario] " << e.what() << std::endl; }
        }

        all_results.push_back(current_test);
    } // Fim do loop principal

    // --- Tabela Final de Resultados ---
    std::cout <<
            "\n\n======================================================================================================================================"
            << std::endl;
    std::cout << "                                           RELATORIO COMPARATIVO FINAL" << std::endl;
    std::cout <<
            "======================================================================================================================================"
            << std::endl;
    std::cout << std::left
            << std::setw(22) << "Teste"
            << std::setw(26) << "Metodo"
            << std::setw(15) << "RAM (MB)"
            << std::setw(20) << "T. Carga (ms)"
            << std::setw(20) << "T. Solver (ms)"
            << std::setw(12) << "Iteracoes"
            << std::setw(15) << "Erro Final" // ||r|| final
            << std::setw(15) << "Epsilon Final" // Epsilon que parou ou último
            << std::setw(15) << "Convergiu (Eps)" // Sim/Nao
            << std::setw(15) << "vs Baseline" // Speedup vs Esparso Bin
            << std::endl;
    std::cout <<
            "--------------------------------------------------------------------------------------------------------------------------------------"
            << std::endl;

    for (const auto &test_result: all_results) {
        const auto &test_name = test_result.test_name;
        const auto &dense_text = test_result.dense_text;
        const auto &dense_binary = test_result.dense_binary;
        const auto &sparse_text = test_result.sparse_text;
        const auto &sparse_binary = test_result.sparse_binary;

        const auto &baseline = sparse_binary; // Baseline é Esparso Binário
        bool baseline_valid = baseline.solve_time_ms > 0;

        auto print_metric = [&](const std::string &label, const PerformanceMetrics &m, const bool is_baseline) {
            if (m.load_time_ms <= 0 && m.solve_time_ms <= 0 && m.estimated_ram_mb <= 0) return;

            std::cout << std::left << std::setw(22) << (label == "Denso / Texto" ? test_name : "")
                    << std::setw(26) << label
                    << std::fixed << std::setprecision(2) << std::setw(15) << m.estimated_ram_mb
                    << std::fixed << std::setprecision(2) << std::setw(20) << m.load_time_ms
                    << std::fixed << std::setprecision(2) << std::setw(20) << m.solve_time_ms
                    << std::setw(12) << m.iterations
                    << std::scientific << std::setprecision(3) << std::setw(14) << m.final_error
                    << std::scientific << std::setprecision(3) << std::setw(14) << m.final_epsilon
                    << std::setw(15) << (m.converged ? "Sim" : "Nao (MaxIt)");


            if (is_baseline) {
                std::cout << std::setw(14) << std::right << "BASELINE";
            } else if (baseline_valid && m.solve_time_ms > 0) {
                const double speedup = baseline.solve_time_ms / m.solve_time_ms;
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << speedup << "x";
                std::cout << std::setw(14) << std::right << ss.str();
            } else {
                std::cout << std::setw(14) << std::right << " N/A";
            }
            std::cout << std::endl;
        };

        if (run_dense) { // Usa a variável 'run_dense' declarada acima
            print_metric("Denso / Texto", dense_text, false);
            print_metric("Denso / Binario", dense_binary, false);
        }
        if (run_sparse) { // Usa a variável 'run_sparse' declarada acima
            print_metric("Esparso / Texto", sparse_text, false);
            print_metric("Esparso / Binario", sparse_binary, true); // Baseline
        }

        std::cout <<
                "--------------------------------------------------------------------------------------------------------------------------------------"
                << std::endl;
    }


    std::cout << "\nBenchmark concluido." << std::endl;

    return 0;
}