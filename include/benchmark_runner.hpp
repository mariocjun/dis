#ifndef ULTRASOUNDBENCHMARK_BENCHMARK_RUNNER_HPP
#define ULTRASOUNDBENCHMARK_BENCHMARK_RUNNER_HPP

#include "types.hpp"   // Para Config, BenchmarkResults, etc.
#include "config.hpp"  // Para a classe Config

/**
 * @brief Executa os pipelines de benchmark definidos na configuração.
 * Carrega dados, chama os solvers apropriados, coleta métricas e
 * preenche a estrutura BenchmarkResults.
 *
 * @param config Objeto Config contendo toda a configuração lida do YAML.
 * @return BenchmarkResults Estrutura contendo os resultados de todos os testes executados.
 */
BenchmarkResults run_benchmarks(const Config& config);


#endif //ULTRASOUNDBENCHMARK_BENCHMARK_RUNNER_HPP