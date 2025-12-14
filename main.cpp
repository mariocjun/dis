#include <algorithm> // Para std::replace
#include <chrono> // Para timing (embora a maior parte esteja no header agora)
#include <filesystem> // Para manipulação de caminhos (C++17)
#include <iomanip>    // Para formatação da tabela
#include <iostream>
#include <omp.h>     // Para OpenMP info
#include <sstream>   // Para formatar speedup
#include <stdexcept> // Para std::runtime_error
#include <string>
#include <vector>

// Inclui o header com a lógica de comparação e funções auxiliares/solvers
#include "include/config.hpp"
#include "include/types.hpp"
#include "solver_comparison.hpp"

// Removida a função create_output_directories daqui pois já existe em
// solver_comparison.hpp

// --- Função Principal (SIMPLIFICADA) ---
int main(int argc, char *argv[]) {
  std::cout << "======================================================"
            << std::endl;
  std::cout << " Comparativo: CGNR Standard vs Pre-condicionado" << std::endl;
  std::cout << "======================================================"
            << std::endl;

  // --- Config Loading Logic ---
  std::string config_file_path = "config.yaml"; // Default
  std::string cmd_output_dir = "";
  if (argc > 2) {
    cmd_output_dir = argv[2];
  }

  // 1. Check if user provided a path via command line
  if (argc > 1) {
    config_file_path = argv[1];
  } else {
    // 2. Try to find config.yaml relative to executable or current path
    std::filesystem::path exe_path(argv[0]);
    std::filesystem::path search_paths[] = {
        std::filesystem::current_path() / "config.yaml",
        exe_path.parent_path() / "config.yaml",
        exe_path.parent_path().parent_path() / "config.yaml",
        exe_path.parent_path().parent_path().parent_path() / "config.yaml"};

    bool found = false;
    for (const auto &p : search_paths) {
      if (std::filesystem::exists(p)) {
        config_file_path = p.string();
        found = true;
        break;
      }
    }

    if (!found) {
      std::cout << "[AVISO] config.yaml nao encontrado nos caminhos padrao. "
                   "Tentando 'config.yaml' no diretorio atual."
                << std::endl;
    }
  }

  std::cout << "[INFO] Tentando carregar configuracao de: " << config_file_path
            << std::endl;

  std::filesystem::path config_path(config_file_path);
  Config config;
  try {
    config = load_config(config_file_path);
  } catch (const std::exception &e) {
    std::cerr << "[ERRO FATAL] Erro ao carregar configuracao: " << e.what()
              << std::endl;
    return 1;
  }

  if (config.run_pipelines.empty()) {
    std::cerr << "[ERRO] Nenhum pipeline definido no arquivo de configuração."
              << std::endl;
    return 1;
  }

  // Pega apenas o primeiro pipeline (Sparse_Comparison)
  const auto &pipeline = config.run_pipelines[0];

  // Prepara o vetor de testes baseado no pipeline
  std::cout << "\n[INFO] Pipeline selecionado: " << pipeline.name << std::endl;
  std::cout << "[INFO] Datasets configurados: ";
  for (const auto &name : pipeline.dataset_names) {
    std::cout << name << " ";
  }
  std::cout << std::endl;

  std::vector<DatasetConfig> tests;
  for (const auto &dataset_name : pipeline.dataset_names) {
    std::cout << "[INFO] Processando dataset: " << dataset_name << std::endl;
    auto it = config.dataset_map.find(dataset_name);
    if (it != config.dataset_map.end()) {
      DatasetConfig dataset_config = *it->second; // Make a copy to modify paths
      std::filesystem::path config_dir = config_path.parent_path();
      dataset_config.h_matrix_csv =
          (config_dir / dataset_config.h_matrix_csv).string();
      dataset_config.g_signal_csv =
          (config_dir / dataset_config.g_signal_csv).string();
      tests.push_back(dataset_config);
      std::cout << "[INFO] Dataset " << dataset_name
                << " adicionado para processamento" << std::endl;
    } else {
      std::cout << "[AVISO] Dataset " << dataset_name
                << " nao encontrado no mapa de configuracoes" << std::endl;
    }
  }

  std::cout << "[INFO] Total de datasets a serem processados: " << tests.size()
            << std::endl;

  // Configuração OpenMP
  int num_threads_to_use = config.settings.num_omp_threads;
  if (num_threads_to_use <= 0) {
    num_threads_to_use = omp_get_max_threads();
  }

  Eigen::setNbThreads(num_threads_to_use);
  omp_set_num_threads(num_threads_to_use);

  std::cout << "\n[INFO] Usando " << num_threads_to_use
            << " threads para os calculos Eigen e OpenMP (Max disponivel: "
            << omp_get_max_threads() << ").\n"
            << std::endl;

  // --- Pré-processamento (Apenas garante que o .sparse.bin existe) ---
  std::cout << "[INFO] Verificando/Criando arquivos binarios esparsos..."
            << std::endl;
  for (const auto &config : tests) {
    std::filesystem::path h_path =
        config.h_matrix_csv; // Correção aqui: usa h_matrix_csv
    if (!std::filesystem::exists(h_path)) {
      std::cerr << "[ERRO] Arquivo CSV da matriz H nao encontrado: "
                << config.h_matrix_csv << std::endl;
      return 1;
    }
    std::filesystem::path data_dir = h_path.parent_path();
    std::filesystem::path sparse_bin_fs_path =
        data_dir / (h_path.filename().string() + ".sparse.bin");
    std::string sparse_bin_path = sparse_bin_fs_path.string();

    if (!std::filesystem::exists(sparse_bin_fs_path)) {
      std::cout << "[AVISO] Criando arquivo binario esparso para "
                << config.h_matrix_csv << " em " << sparse_bin_path << "..."
                << std::endl;
      try {
        saveSparseMatrix(
            convertCsvToSparse(config.h_matrix_csv,
                               config.image_rows * config.image_cols),
            sparse_bin_path);
        std::cout << "[SUCESSO] Arquivo binario esparso criado: "
                  << sparse_bin_path << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "[ERRO] Falha ao criar arquivo binario esparso: "
                  << e.what() << std::endl;
        return 1;
      }
    }
  }
  std::cout << "[INFO] Pre-processamento concluido.\n" << std::endl;

  // Padronizar o caminho de saída para corresponder ao que o Python espera
  std::filesystem::path output_dir;
  if (!cmd_output_dir.empty()) {
    output_dir = cmd_output_dir;
    config.settings.output_base_dir =
        cmd_output_dir; // Update settings used by solvers
  } else {
    output_dir = config.settings.output_base_dir;
  }
  try {
    create_output_directories(
        output_dir); // Usa a função do solver_comparison.hpp
    std::cout << "[INFO] Estrutura de diretorios criada: "
              << std::filesystem::absolute(output_dir) << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "[ERRO] Nao foi possivel criar os diretorios de saida: "
              << e.what() << std::endl;
    return 1;
  }

  // --- Loop Principal de Testes ---
  std::vector<std::tuple<std::string, PerformanceMetrics, PerformanceMetrics>>
      all_results;

  // Verificação adicional para garantir que estamos processando apenas os
  // datasets selecionados
  std::cout << "\n[INFO] Verificando datasets a serem processados:"
            << std::endl;
  for (const auto &test_config : tests) {
    std::cout << "  - Dataset: " << test_config.name << " ("
              << test_config.description << ")" << std::endl;
    bool is_in_pipeline =
        std::find(pipeline.dataset_names.begin(), pipeline.dataset_names.end(),
                  test_config.name) != pipeline.dataset_names.end();
    if (!is_in_pipeline) {
      std::cout << "[AVISO] Dataset " << test_config.name
                << " nao esta no pipeline atual. Ignorando." << std::endl;
      continue;
    }

    std::cout << "\n========================================\nINICIANDO TESTE: "
              << test_config.description <<
        // Usa .description
        "\n========================================" << std::endl;

    try {
      auto [std_metrics, precond_metrics] =
          run_sparse_comparison(test_config, config.settings);
      all_results.push_back(std::make_tuple(test_config.description,
                                            std_metrics, precond_metrics));
      std::cout << "Teste " << test_config.description
                << " concluido com sucesso." << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[ERRO FATAL] Falha ao executar comparacao para o teste "
                << test_config.description << ": " << e.what() << std::endl;
    }
  }

  // --- Tabela Final de Resultados ---
  std::cout << "\n\n==========================================================="
               "==============================================================="
               "============================"
            << std::endl;
  std::cout << "                                                    RELATORIO "
               "COMPARATIVO FINAL"
            << std::endl;
  std::cout << "==============================================================="
               "==============================================================="
               "========================"
            << std::endl;
  std::cout << std::left << std::setw(22) << "Teste" << std::setw(28)
            << "Metodo" << std::setw(15) << "RAM (MB)" << std::setw(20)
            << "T. Carga (ms)" << std::setw(20) << "T. Solver (ms)"
            << std::setw(12) << "Iteracoes" << std::setw(15) << "Erro Final"
            << std::setw(15) << "Epsilon Final" << std::setw(15) << "Convergiu"
            << std::setw(15) << "Speedup" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------------"
               "------------------------"
            << std::endl;

  for (const auto &[test_name, std_metrics, precond_metrics] : all_results) {
    auto print_metric = [&](const PerformanceMetrics &m,
                            const std::string &method_name, bool is_baseline) {
      if (m.load_time_ms <= 0 && m.solve_time_ms <= 0) {
        std::cout << std::left << std::setw(22)
                  << (is_baseline ? test_name : "") << std::setw(28)
                  << method_name << std::setw(15 + 20 + 20 + 12 + 15 + 15 + 15)
                  << " [FALHOU]" << std::setw(14) << std::right
                  << (is_baseline ? "BASELINE" : "N/A") << std::endl;
        return;
      }

      std::cout << std::left << std::setw(22) << (is_baseline ? test_name : "")
                << std::setw(28) << method_name << std::fixed
                << std::setprecision(2) << std::setw(15) << m.estimated_ram_mb
                << std::fixed << std::setprecision(2) << std::setw(20)
                << m.load_time_ms << std::fixed << std::setprecision(2)
                << std::setw(20) << m.solve_time_ms << std::setw(12)
                << m.iterations << std::scientific << std::setprecision(3)
                << std::setw(14) << m.final_error << std::scientific
                << std::setprecision(3) << std::setw(14) << m.final_epsilon
                << std::setw(15) << (m.converged ? "Sim" : "Nao (MaxIt)");

      if (is_baseline) {
        std::cout << std::setw(14) << std::right << "BASELINE";
      } else if (std_metrics.solve_time_ms > 0 && m.solve_time_ms > 0) {
        const double speedup = std_metrics.solve_time_ms / m.solve_time_ms;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << speedup << "x";
        std::cout << std::setw(14) << std::right << ss.str();
      } else {
        std::cout << std::setw(14) << std::right << "N/A";
      }
      std::cout << std::endl;
    };

    print_metric(std_metrics, "CGNR Standard", true);
    print_metric(precond_metrics, "CGNR Pre-condicionado", false);

    std::cout << "-------------------------------------------------------------"
                 "-------------------------------------------------------------"
                 "----------------------------"
              << std::endl;
  }

  std::cout << "\nBenchmark concluido." << std::endl;
  std::cout << "[INFO] Para visualizar os graficos de convergencia, execute:"
            << std::endl;
  std::cout << "[INFO] python scripts/plot_convergence.py ../output_csv/metrics"
            << std::endl;
  std::cout
      << "[INFO] Para gerar animacoes da reconstrucao das imagens, execute:"
      << std::endl;
  std::cout
      << "[INFO] python scripts/visualize_iterations.py ../output_csv/images"
      << std::endl;
  return 0;
}
