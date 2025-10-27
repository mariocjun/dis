#ifndef ULTRASOUNDBENCHMARK_REPORTING_HPP
#define ULTRASOUNDBENCHMARK_REPORTING_HPP

#include "types.hpp"   // Para BenchmarkResults, Config
#include "config.hpp"  // Para Config
#include <string>

/**
 * @brief Gera a tabela de relatório comparativo final.
 * Imprime a tabela no console e, opcionalmente, salva em um arquivo.
 *
 * @param benchmark_results Estrutura contendo os resultados de todos os testes.
 * @param config Objeto Config contendo a configuração global e dos métodos/datasets.
 */
void generate_report(const BenchmarkResults& benchmark_results, const Config& config);

#endif //ULTRASOUNDBENCHMARK_REPORTING_HPP