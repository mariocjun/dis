#include "../include/io_utils.hpp" // Inclui as declarações
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm> // Para std::remove_if, std::all_of
#include <cmath>     // Para std::abs, std::sqrt
#include <fstream>
#include <iomanip> // Para std::scientific, std::setprecision
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// --- Implementações das Funções de Carregamento ---

Eigen::VectorXd loadVectorData(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
  std::vector<double> values;
  std::string line;
  int line_num = 0;
  while (std::getline(file, line)) {
    line_num++;
    // C++17 compatível:
    if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace))
      continue;
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      try {
        // C++17 compatível:
        cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace),
                   cell.end());
        if (!cell.empty()) {
          values.push_back(std::stod(cell));
        }
      } catch (const std::invalid_argument &) {
        std::cerr << "[AVISO] Ignorando valor nao numerico em: " << path
                  << ", linha: " << line_num << ", celula: '" << cell << "'"
                  << std::endl;
      } catch (const std::out_of_range &) {
        std::cerr << "[AVISO] Ignorando valor fora do range em: " << path
                  << ", linha: " << line_num << ", celula: '" << cell << "'"
                  << std::endl;
      }
    }
  }
  if (values.empty()) {
    throw std::runtime_error("Nenhum dado numerico valido encontrado em: " +
                             path);
  }
  Eigen::Map<Eigen::VectorXd> vec_map(values.data(), values.size());
  return Eigen::VectorXd(vec_map);
}

Eigen::MatrixXd loadDenseData(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
  std::vector<double> values;
  std::string line;
  int rows = 0;
  long long cols = -1;
  int line_num = 0;

  while (std::getline(file, line)) {
    line_num++;
    // C++17 compatível:
    if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace))
      continue;

    std::stringstream lineStream(line);
    std::string cell;
    long long current_cols = 0;
    std::vector<double> row_values;

    while (std::getline(lineStream, cell, ',')) {
      try {
        // C++17 compatível:
        cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace),
                   cell.end());
        if (!cell.empty()) {
          row_values.push_back(std::stod(cell));
        } else {
          row_values.push_back(0.0);
          std::cerr << "[AVISO] Celula vazia encontrada em: " << path
                    << ", linha: " << line_num
                    << ", coluna: " << current_cols + 1 << ". Assumindo 0.0."
                    << std::endl;
        }
        current_cols++;
      } catch (const std::invalid_argument &) {
        std::cerr << "[AVISO] Ignorando valor nao numerico em: " << path
                  << ", linha: " << line_num << ", celula: '" << cell << "'"
                  << std::endl;
        row_values.push_back(0.0);
        current_cols++;
      } catch (const std::out_of_range &) {
        std::cerr << "[AVISO] Ignorando valor fora do range em: " << path
                  << ", linha: " << line_num << ", celula: '" << cell << "'"
                  << std::endl;
        row_values.push_back(0.0);
        current_cols++;
      }
    }

    if (cols == -1) {
      if (current_cols == 0)
        continue;
      cols = current_cols;
    } else if (current_cols != cols) {
      std::cerr << "[ERRO] Numero inconsistente de colunas na linha "
                << line_num << " do arquivo " << path << ". Esperado: " << cols
                << ", Encontrado: " << current_cols << ". Abortando."
                << std::endl;
      throw std::runtime_error("Inconsistencia de colunas no CSV denso.");
    }

    values.insert(values.end(), row_values.begin(), row_values.end());
    rows++;
  }
  if (rows == 0 || cols <= 0 || values.empty()) {
    throw std::runtime_error(
        "Nao foi possivel carregar dados validos da matriz densa de: " + path +
        " (rows=" + std::to_string(rows) + ", cols=" + std::to_string(cols) +
        ")");
  }

  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      mat_map(values.data(), rows, cols);
  return Eigen::MatrixXd(mat_map);
}

Eigen::MatrixXd loadDenseMatrix(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " +
                             path);
  Eigen::Index rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  if (!file || file.eof())
    throw std::runtime_error(
        "Erro ao ler cabecalho do arquivo binario denso: " + path);
  if (rows <= 0 || cols <= 0)
    throw std::runtime_error("Dimensoes invalidas (" + std::to_string(rows) +
                             "x" + std::to_string(cols) +
                             ") lidas do arquivo binario denso: " + path);

  Eigen::MatrixXd mat(rows, cols);
  file.read(reinterpret_cast<char *>(mat.data()),
            static_cast<std::streamsize>(rows) * cols * sizeof(double));
  if (!file) {
    if (file.eof() &&
        (static_cast<std::streamsize>(rows) * cols * sizeof(double) > 0)) {
      throw std::runtime_error(
          "Fim de arquivo inesperado ao ler dados do arquivo binario denso: " +
          path);
    } else if (!file.eof()) {
      throw std::runtime_error(
          "Erro de leitura ao processar dados do arquivo binario denso: " +
          path);
    }
  }
  return mat;
}

Eigen::SparseMatrix<double> convertCsvToSparse(const std::string &path,
                                               int expected_cols) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
  std::vector<Eigen::Triplet<double>> tripletList;
  std::string line;
  int row = 0;
  long long actual_cols = -1;
  int line_num = 0;

  while (std::getline(file, line)) {
    line_num++;
    // C++17 compatível:
    if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace))
      continue;

    std::stringstream lineStream(line);
    std::string cell;
    int col = 0;
    while (std::getline(lineStream, cell, ',')) {
      try {
        // C++17 compatível:
        cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace),
                   cell.end());
        if (!cell.empty()) {
          double value = std::stod(cell);
          if (std::abs(value) > 1e-12) {
            tripletList.emplace_back(row, col, value);
          }
        }
      } catch (const std::invalid_argument &) {
        std::cerr << "[AVISO] Ignorando valor nao numerico em CSV esparso: "
                  << path << ", linha: " << line_num << ", celula: '" << cell
                  << "'" << std::endl;
      } catch (const std::out_of_range &) {
        std::cerr << "[AVISO] Ignorando valor fora do range em CSV esparso: "
                  << path << ", linha: " << line_num << ", celula: '" << cell
                  << "'" << std::endl;
      }
      col++;
    }
    if (actual_cols == -1) {
      if (col == 0)
        continue;
      actual_cols = col;
    } else if (col != actual_cols) {
      std::cerr << "[ERRO] Numero inconsistente de colunas na linha "
                << line_num << " do arquivo esparso " << path
                << ". Esperado: " << actual_cols << ", Encontrado: " << col
                << ". Abortando." << std::endl;
      throw std::runtime_error("Inconsistencia de colunas no CSV esparso.");
    }
    row++;
  }
  if (row == 0 || actual_cols <= 0) {
    throw std::runtime_error("Nao foi possivel ler nenhuma linha/coluna valida "
                             "do arquivo esparso: " +
                             path);
  }
  if (expected_cols > 0 && actual_cols != expected_cols) {
    std::cerr << "[AVISO] Numero de colunas lido (" << actual_cols
              << ") difere do esperado (" << expected_cols << ") para " << path
              << ". Usando o numero lido." << std::endl;
  }

  Eigen::SparseMatrix<double> mat(row, actual_cols);
  if (!tripletList.empty()) {
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
  } else {
    std::cerr
        << "[AVISO] Nenhum elemento nao-zero (acima de 1e-12) encontrado em "
        << path << std::endl;
  }
  mat.makeCompressed();
  return mat;
}

Eigen::SparseMatrix<double> loadSparseMatrix(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel abrir o arquivo binario: " +
                             path);
  Eigen::Index rows, cols;
  Eigen::Index nonZeros;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char *>(&nonZeros), sizeof(nonZeros));

  if (!file || file.eof())
    throw std::runtime_error(
        "Erro ao ler cabecalho do arquivo binario esparso: " + path);
  if (rows < 0 || cols < 0 || nonZeros < 0)
    throw std::runtime_error("Dimensoes invalidas (" + std::to_string(rows) +
                             "x" + std::to_string(cols) +
                             ", nnz=" + std::to_string(nonZeros) +
                             ") lidas do arquivo binario esparso: " + path);

  Eigen::SparseMatrix<double> mat(rows, cols);
  mat.makeCompressed();
  mat.resizeNonZeros(nonZeros);

  if (nonZeros > 0) {
    if (!mat.valuePtr() || !mat.innerIndexPtr()) {
      throw std::runtime_error(
          "Ponteiros value/inner invalidos apos resizeNonZeros ao carregar: " +
          path);
    }
    file.read(reinterpret_cast<char *>(mat.valuePtr()),
              nonZeros * sizeof(double));
    file.read(reinterpret_cast<char *>(mat.innerIndexPtr()),
              nonZeros * sizeof(int));
  }

  if (!mat.outerIndexPtr()) {
    throw std::runtime_error(
        "Ponteiro outerIndex invalido apos makeCompressed ao carregar: " +
        path);
  }
  file.read(reinterpret_cast<char *>(mat.outerIndexPtr()),
            (mat.outerSize() + 1) * sizeof(int));

  if (!file) {
    if (file.eof()) {
      std::cerr << "[AVISO] Fim de arquivo prematuro ao ler dados de " << path
                << ". A matriz pode estar incompleta." << std::endl;
      throw std::runtime_error("Fim de arquivo inesperado ao ler dados do "
                               "arquivo binario esparso: " +
                               path);
    } else {
      throw std::runtime_error(
          "Erro de leitura ao processar dados do arquivo binario esparso: " +
          path);
    }
  }
  mat.finalize();
  return mat;
}

// --- Implementações das Funções de Salvamento ---

void saveDenseMatrix(const Eigen::MatrixXd &mat, const std::string &path) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel criar o arquivo binario: " +
                             path);
  const auto rows = mat.rows(), cols = mat.cols();
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char *>(mat.data()),
             static_cast<std::streamsize>(rows) * cols * sizeof(double));
  if (!file)
    throw std::runtime_error("Erro ao escrever no arquivo binario denso: " +
                             path);
  file.close();
  if (!file)
    throw std::runtime_error("Erro ao fechar o arquivo binario denso: " + path);
}

void saveSparseMatrix(const Eigen::SparseMatrix<double> &mat,
                      const std::string &path) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open())
    throw std::runtime_error("Nao foi possivel criar o arquivo binario: " +
                             path);

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
      throw std::runtime_error(
          "Ponteiros internos invalidos ao salvar matriz esparsa para: " +
          path);
    }
    file.write(reinterpret_cast<const char *>(compressed_mat.valuePtr()),
               nonZeros * sizeof(double));
    file.write(reinterpret_cast<const char *>(compressed_mat.innerIndexPtr()),
               nonZeros * sizeof(int));
  }
  if (!compressed_mat.outerIndexPtr()) {
    throw std::runtime_error(
        "Ponteiro outerIndex invalido ao salvar matriz esparsa para: " + path);
  }
  file.write(reinterpret_cast<const char *>(compressed_mat.outerIndexPtr()),
             (outerSize + 1) * sizeof(int));

  if (!file)
    throw std::runtime_error("Erro ao escrever no arquivo binario esparso: " +
                             path);
  file.close();
  if (!file)
    throw std::runtime_error("Erro ao fechar o arquivo binario esparso: " +
                             path);
}

void saveHistoryToCSV(const std::vector<double> &history,
                      const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "[AVISO] Nao foi possivel criar o arquivo de historico: "
              << filename << std::endl;
    return;
  }
  file << "Iteration,ResidualNorm\n";
  for (size_t i = 0; i < history.size(); ++i) {
    file << i + 1 << "," << std::scientific << std::setprecision(8)
         << history[i] << "\n";
  }
  if (!file) {
    std::cerr << "[AVISO] Erro ao escrever no arquivo de historico: "
              << filename << std::endl;
  } else {
    // Removido cout daqui para evitar poluir a saída do benchmark
    // std::cout << "[INFO] Historico de convergencia salvo em: " << filename <<
    // std::endl;
  }
  file.close();
}

// Função auxiliar para manter compatibilidade com ReconstructionResult
void saveHistoryToCSV(const ReconstructionResult &result,
                      const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "[AVISO] Nao foi possivel criar o arquivo de historico: "
              << filename << std::endl;
    return;
  }

  // Cabeçalho com mais métricas
  file << "Iteration,ResidualNorm,SolutionNorm,ExecutionTime_ms\n";

  for (size_t i = 0; i < result.residual_history.size(); ++i) {
    file << i + 1 << ", " << std::scientific << std::setprecision(8)
         << result.residual_history[i] << ", " << std::scientific
         << std::setprecision(8)
         << (i < result.solution_history.size() ? result.solution_history[i]
                                                : 0.0)
         << ", " << std::fixed << std::setprecision(2)
         << result.execution_time_ms << "\n";
  }

  if (!file) {
    std::cerr << "[AVISO] Erro ao escrever no arquivo de historico: "
              << filename << std::endl;
  } else {
    std::cout << "[INFO] Historico de convergencia salvo em: " << filename
              << std::endl;
  }
  file.close();
}

void saveLcurveToCSV(const ReconstructionResult &result,
                     const std::string &filename) {
  if (result.residual_history.size() != result.solution_history.size()) {
    std::cerr
        << "[AVISO] Tamanhos incompativeis (" << result.residual_history.size()
        << " vs " << result.solution_history.size()
        << ") de historico de residuo e solucao para L-curve. Nao salvando "
        << filename << std::endl;
    return;
  }
  if (result.residual_history.empty()) {
    // std::cerr << "[AVISO] Historico vazio, nada para salvar em L-curve: " <<
    // filename << std::endl; // Opcional
    return;
  }
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "[AVISO] Nao foi possivel criar o arquivo de cotovelo: "
              << filename << std::endl;
    return;
  }
  file << "Iteration,SolutionNorm,ResidualNorm\n";
  for (size_t i = 0; i < result.residual_history.size(); ++i) {
    file << i + 1 << "," << std::scientific << std::setprecision(8)
         << result.solution_history[i] << "," << std::scientific
         << std::setprecision(8) << result.residual_history[i] << "\n";
  }
  if (!file) {
    std::cerr << "[AVISO] Erro ao escrever no arquivo L-curve: " << filename
              << std::endl;
  } else {
    // Removido cout daqui
    // std::cout << "[INFO] Dados L-curve salvos em: " << filename << std::endl;
  }
  file.close();
}

void saveImageVectorToCsv(const Eigen::VectorXd &vec,
                          const std::string &filename, int img_rows,
                          int img_cols) {
  if (vec.size() != static_cast<long long>(img_rows) * img_cols) {
    std::cerr << "[AVISO] Tamanho do vetor (" << vec.size()
              << ") nao corresponde as dimensoes da imagem (" << img_rows << "x"
              << img_cols << "). Nao salvando imagem: " << filename
              << std::endl;
    return;
  }
  if (img_rows <= 0 || img_cols <= 0) {
    std::cerr << "[AVISO] Dimensoes invalidas da imagem (" << img_rows << "x"
              << img_cols << "). Nao salvando imagem: " << filename
              << std::endl;
    return;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "[AVISO] Nao foi possivel criar o arquivo CSV da imagem: "
              << filename << std::endl;
    return;
  }

  file << std::scientific << std::setprecision(8);

  for (int i = 0; i < img_rows; ++i) {
    for (int j = 0; j < img_cols; ++j) {
      long long index = static_cast<long long>(i) * img_cols +
                        j; // RowMajor Indexing (same as Python reshape)
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
    std::cerr << "[AVISO] Erro ao escrever no arquivo CSV da imagem: "
              << filename << std::endl;
  } else {
    // Removido cout daqui
    // std::cout << "[INFO] Imagem reconstruida salva em: " << filename <<
    // std::endl;
  }
  file.close();
}

void saveImageMetadata(const std::string &filename,
                       const std::map<std::string, std::string> &metadata) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "[AVISO] Nao foi possivel criar o arquivo de metadados: "
              << filename << std::endl;
    return;
  }
  std::cout << "[INFO] Metadata: ";
  for (const auto &[key, value] : metadata) {
    file << key << ": " << value << "\n";
    std::cout << "[" << key << ": " << value << "] ";
  }
  std::cout << std::endl;
  if (!file) {
    std::cerr << "[AVISO] Erro ao escrever no arquivo de metadados: "
              << filename << std::endl;
  }
  file.close();
}