#ifndef ULTRASOUNDBENCHMARK_REPORTING_HPP
#define ULTRASOUNDBENCHMARK_REPORTING_HPP

#include "config.hpp" // Para Config
#include "types.hpp"  // Para BenchmarkResults, Config
#include <string>


/**
 * @brief Gera a tabela de relatório comparativo final.
 * Imprime a tabela no console e, opcionalmente, salva em um arquivo.
 *
 * @param benchmark_results Estrutura contendo os resultados de todos os testes.
 * @param config Objeto Config contendo a configuração global e dos
 * métodos/datasets.
 */
void generate_report(const BenchmarkResults &benchmark_results,
                     const Config &config);

/**
 * @brief Prints a detailed breakdown of estimated memory usage.
 */
#include <Eigen/SparseCore>
void printMemoryBreakdown(const Eigen::SparseMatrix<double> &H,
                          const Eigen::VectorXd &g);

#endif // ULTRASOUNDBENCHMARK_REPORTING_HPP