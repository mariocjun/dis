#include "../include/reporting.hpp" // Inclui a declaração da função
#include "../include/config.hpp"    // Para acessar descrições, etc.
#include "../include/types.hpp"     // Para as structs de dados
#include <filesystem> // Para construir o caminho do arquivo de relatório
#include <fstream>    // Para salvar o relatório em arquivo (opcional)
#include <iomanip>    // Para formatação (setw, setprecision, etc.)
#include <iostream>
#include <map>
#include <sstream> // Para formatar o speedup
#include <string>
#include <vector>

// --- Implementação da Geração do Relatório ---

void generate_report(const BenchmarkResults &benchmark_results,
                     const Config &config) {
  // String stream para construir o relatório (para console e arquivo)
  std::stringstream report_ss;

  // --- Cabeçalho do Relatório ---
  report_ss << "\n\n==========================================================="
               "==============================================================="
               "============================"
            << std::endl;
  report_ss << "                                           RELATORIO "
               "COMPARATIVO FINAL (ESPARSO)"
            << std::endl;
  report_ss << "==============================================================="
               "==============================================================="
               "========================"
            << std::endl;
  report_ss << std::left << std::setw(22) << "Teste" << std::setw(28)
            << "Metodo Otimizado" << std::setw(15) << "RAM (MB)"
            << std::setw(20) << "T. Carga (ms)" << std::setw(20)
            << "T. Solver (ms)" << std::setw(12) << "Iteracoes" << std::setw(15)
            << "Erro Final" << std::setw(15) << "Epsilon Final" << std::setw(15)
            << "Convergiu (Eps)" << std::setw(15)
            << "vs Standard" // Comparação vs standard (baseline)
            << std::endl;
  report_ss << "---------------------------------------------------------------"
               "---------------------------------------------------------------"
               "------------------------"
            << std::endl;

  // --- Itera sobre os resultados por dataset ---
  // Usamos config.datasets para manter a ordem original
  for (const auto &dataset_config : config.datasets) {
    const std::string &dataset_name = dataset_config.name;

    // Verifica se há resultados para este dataset
    auto dataset_results_it = benchmark_results.results.find(dataset_name);
    if (dataset_results_it == benchmark_results.results.end()) {
      continue; // Pula se nenhum método rodou para este dataset
    }
    const auto &method_results_map = dataset_results_it->second;

    // Encontra as métricas do baseline (standard) para este dataset, se
    // existirem
    const PerformanceMetrics *baseline_metrics_ptr = nullptr;
    std::string baseline_method_name =
        "sparse_standard"; // Assume que este é o baseline
    auto baseline_it = method_results_map.find(baseline_method_name);
    if (baseline_it != method_results_map.end()) {
      baseline_metrics_ptr = &baseline_it->second;
    }
    bool baseline_valid =
        baseline_metrics_ptr && baseline_metrics_ptr->solve_time_ms > 0;

    bool first_method_for_dataset =
        true; // Para imprimir o nome do teste só uma vez

    // Itera sobre os métodos na ordem definida na config, para consistência
    for (const auto &method_config : config.methods) {
      // Verifica se este método foi executado para este dataset
      auto metrics_it = method_results_map.find(method_config.name);
      if (metrics_it == method_results_map.end()) {
        continue; // Pula se este método não rodou para este dataset
      }
      const PerformanceMetrics &m = metrics_it->second;
      bool is_baseline = (method_config.name == baseline_method_name);

      // Imprime a linha da métrica
      // Verifica se houve erro (métricas zeradas na função de comparação)
      if (m.load_time_ms <= 0 && m.solve_time_ms <= 0 &&
          m.estimated_ram_mb <= 0 && m.iterations == 0 &&
          m.optimization_type != "none") {
        report_ss << std::left << std::setw(22)
                  << (first_method_for_dataset ? dataset_config.description
                                               : "") // Nome do Teste
                  << std::setw(28)
                  << method_config.description // Nome do Método
                  << std::setw(15 + 20 + 20 + 12 + 15 + 15 + 15)
                  << " [FALHOU]" // Mensagem de Falha
                  << std::setw(14) << std::right
                  << (is_baseline ? "BASELINE" : " N/A") // Speedup
                  << std::endl;
      } else if (m.optimization_type !=
                 "none") { // Imprime apenas se as métricas foram preenchidas
        report_ss << std::left << std::setw(22)
                  << (first_method_for_dataset ? dataset_config.description
                                               : "") // Nome do Teste
                  << std::setw(28)
                  << method_config.description // Nome do Método
                  << std::fixed << std::setprecision(2) << std::setw(15)
                  << m.estimated_ram_mb << std::fixed << std::setprecision(2)
                  << std::setw(20) << m.load_time_ms << std::fixed
                  << std::setprecision(2) << std::setw(20) << m.solve_time_ms
                  << std::setw(12) << m.iterations << std::scientific
                  << std::setprecision(3) << std::setw(14) << m.final_error
                  << std::scientific << std::setprecision(3) << std::setw(14)
                  << m.final_epsilon << std::setw(15)
                  << (m.converged ? "Sim" : "Nao (MaxIt)");

        // Calcula Speedup vs Baseline (Standard)
        if (is_baseline) {
          report_ss << std::setw(14) << std::right << "BASELINE";
        } else if (baseline_valid && m.solve_time_ms > 0) {
          const double speedup =
              baseline_metrics_ptr->solve_time_ms / m.solve_time_ms;
          std::stringstream ss_speedup;
          ss_speedup << std::fixed << std::setprecision(2) << speedup << "x";
          report_ss << std::setw(14) << std::right << ss_speedup.str();
        } else {
          report_ss << std::setw(14) << std::right << " N/A";
        }
        report_ss << std::endl;
      }
      first_method_for_dataset =
          false; // Só imprime nome do teste na primeira linha
    } // Fim loop methods

    // Linha separadora entre datasets
    if (!method_results_map
             .empty()) { // Só imprime se houve resultados para este dataset
      report_ss << "-----------------------------------------------------------"
                   "-----------------------------------------------------------"
                   "--------------------------------"
                << std::endl;
    }
  } // Fim loop datasets

  report_ss << "\nBenchmark concluido." << std::endl;

  // --- Imprime no Console ---
  std::cout << report_ss.str();

  // --- Salva em Arquivo (Opcional) ---
  try {
    std::filesystem::path report_file_path = config.settings.output_base_dir;
    report_file_path /= "summary_report.txt"; // Nome do arquivo de relatório
    std::ofstream report_file(report_file_path);
    if (report_file.is_open()) {
      report_file << report_ss.str();
      report_file.close();
      std::cout << "\n[INFO] Relatorio final salvo em: "
                << std::filesystem::absolute(report_file_path) << std::endl;
    } else {
      std::cerr << "[AVISO] Nao foi possivel abrir o arquivo de relatorio para "
                   "escrita: "
                << report_file_path << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "[AVISO] Erro ao salvar relatorio final em arquivo: "
              << e.what() << std::endl;
  }
}

void printMemoryBreakdown(const Eigen::SparseMatrix<double> &H,
                          const Eigen::VectorXd &g) {
  std::cout << "=== MEMORY BREAKDOWN ===" << std::endl;

  double matrix_mb = (H.nonZeros() * (sizeof(double) + sizeof(int)) +
                      (H.outerSize() + 1) * sizeof(int)) /
                     (1024.0 * 1024.0);
  int vec_size = H.cols();
  // 5 auxiliary vectors for CGNR: r, p, q, z0, x (approximately) + input g
  // Assuming double precision
  double vectors_mb = (6.0 * vec_size * sizeof(double)) / (1024.0 * 1024.0);

  // Simple preconditioner (Jacobi) is diagonal, so roughly 1 vector size
  double precond_mb = (vec_size * sizeof(double)) / (1024.0 * 1024.0);

  // Approximating IO buffer as full dense matrix size is confusing,
  // usually we load sparse. But let's follow user suggestion roughly or
  // fit to reality. User suggested dense size buffer?
  // "double io_buffer_mb = (H.rows() * H.cols() * sizeof(double)) /
  // (1024.0*1024.0);" That's HUGE for sparse matrices. Let's assume IO buffer
  // is proportional to NonZeros for sparse.
  double io_buffer_mb = matrix_mb; // buffer for loading

  double total_mb = matrix_mb + vectors_mb + precond_mb + io_buffer_mb;

  std::cout << "  Matrix H:      " << std::fixed << std::setprecision(2)
            << matrix_mb << " MB" << std::endl;
  std::cout << "  Work vectors:  " << vectors_mb << " MB" << std::endl;
  std::cout << "  Precond:       " << precond_mb << " MB" << std::endl;
  std::cout << "  I/O Buffer:    " << io_buffer_mb << " MB" << std::endl;
  std::cout << "  TOTAL:         " << total_mb << " MB" << std::endl;
}