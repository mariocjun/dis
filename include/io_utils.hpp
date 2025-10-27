#ifndef ULTRASOUNDBENCHMARK_IO_UTILS_HPP
#define ULTRASOUNDBENCHMARK_IO_UTILS_HPP

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "types.hpp" // Para ReconstructionResult

// --- Declarações das Funções de Carregamento ---

// Carrega um vetor de um arquivo CSV (uma coluna ou linha longa)
Eigen::VectorXd loadVectorData(const std::string& path);

// Carrega uma matriz densa de um arquivo CSV
Eigen::MatrixXd loadDenseData(const std::string& path);

// Carrega uma matriz densa de um arquivo binário
Eigen::MatrixXd loadDenseMatrix(const std::string& path);

// Converte um CSV para uma matriz esparsa
Eigen::SparseMatrix<double> convertCsvToSparse(const std::string& path, int expected_cols);

// Carrega uma matriz esparsa de um arquivo binário
Eigen::SparseMatrix<double> loadSparseMatrix(const std::string& path);

// --- Declarações das Funções de Salvamento ---

// Salva uma matriz densa em um arquivo binário
void saveDenseMatrix(const Eigen::MatrixXd& mat, const std::string& path);

// Salva uma matriz esparsa em um arquivo binário
void saveSparseMatrix(const Eigen::SparseMatrix<double>& mat, const std::string& path);

// Salva o histórico de normas de resíduo em CSV
void saveHistoryToCSV(const std::vector<double>& history, const std::string& filename);

// Salva os dados para a L-curve (norma da solução vs norma do resíduo) em CSV
void saveLcurveToCSV(const ReconstructionResult& result, const std::string& filename);

// Salva o vetor da imagem reconstruída (f) em formato CSV (matricial)
void saveImageVectorToCsv(const Eigen::VectorXd& vec, const std::string& filename, int img_rows, int img_cols);


#endif //ULTRASOUNDBENCHMARK_IO_UTILS_HPP