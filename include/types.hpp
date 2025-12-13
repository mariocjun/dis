#ifndef ULTRASOUNDBENCHMARK_TYPES_HPP
#define ULTRASOUNDBENCHMARK_TYPES_HPP

#include <Eigen/Core> // Para Eigen::VectorXd
#include <map>        // Para std::map
#include <string>
#include <vector>

// Estrutura para armazenar o resultado de uma reconstrução
struct ReconstructionResult {
  Eigen::VectorXd image; // Vetor f reconstruído
  int iterations{};
  double final_error{};   // Norma do resíduo final ||r||
  double final_epsilon{}; // Valor final de epsilon = | ||r_i+1|| - ||r_i|| |
  double execution_time_ms{};
  bool converged{}; // Indica se parou pela tolerância epsilon
  std::vector<double> residual_history; // Histórico da norma ||r_i||
  std::vector<double> solution_history; // Histórico da norma ||f_i||
  double precond_setup_time_ms =
      0.0; // [NEW] Tempo alocado para setup (ex: Jacobi)
};

// Estrutura para configurar um conjunto de dados de teste (lido do YAML)
struct DatasetConfig {
  std::string name;         // Nome curto (ex: "60x60_G1")
  std::string description;  // Descrição (ex: "60x60 (Sinal G-1)")
  std::string h_matrix_csv; // Caminho para o CSV da matriz H
  std::string g_signal_csv; // Caminho para o CSV do sinal G
  int image_rows = 0;
  int image_cols = 0;
};

// Estrutura para configurar um método de reconstrução (lido do YAML)
struct MethodConfig {
  std::string name;         // Nome curto (ex: "sparse_standard")
  std::string description;  // Descrição (ex: "Esparso / Binario (Standard)")
  std::string solver;       // Identificador do solver (ex: "cgnr_standard")
  bool use_binary = true;   // Usar .bin em vez de .csv para H?
  bool is_baseline = false; // É a referência para speedup?
};

// Estrutura para configurar um pipeline de execução (lido do YAML)
struct PipelineConfig {
  std::string name;
  std::string description;
  std::vector<std::string> method_names;  // Nomes dos métodos a rodar
  std::vector<std::string> dataset_names; // Nomes dos datasets a usar
};

// Estrutura para armazenar as métricas de desempenho de uma execução
struct PerformanceMetrics {
  std::string optimization_type; // "standard", "jacobi", "fista"
  double load_time_ms = 0.0;
  double solve_time_ms = 0.0;
  double estimated_ram_mb = 0.0; // RAM estimada apenas para H
  int iterations = 0;
  double final_error = 0.0;   // ||r|| final
  double final_epsilon = 0.0; // Epsilon final
  bool converged = false;
  std::vector<double> solution_history; // History for post-analysis
  double precond_setup_time_ms = 0.0;
};

// Estrutura para guardar todos os resultados de um pipeline
struct BenchmarkResults {
  // Mapeia nome do dataset -> (Mapeia nome do método -> Métricas)
  std::map<std::string, std::map<std::string, PerformanceMetrics>> results;
  // Guarda ponteiros para a config original para referência
  const MethodConfig *baseline_method = nullptr;
};

// Estrutura global para as configurações lidas do YAML
struct GlobalSettings {
  std::string output_base_dir = "../output";
  double epsilon_tolerance = 1.0e-4;
  int max_iterations = 10;
  int l_curve_iterations = 50; // Iterações para curva L
  bool save_intermediate_images = false;
  int num_omp_threads = 0; // 0 = padrão
};

#endif // ULTRASOUNDBENCHMARK_TYPES_HPP