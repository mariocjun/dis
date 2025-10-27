#include <iostream>
#include <vector>
#include <string>
#include <filesystem> // Para manipulação de caminhos (C++17)
#include <chrono>     // Para timing (embora a maior parte esteja no header agora)
#include <iomanip>    // Para formatação da tabela
#include <sstream>    // Para formatar speedup
#include <omp.h>      // Para OpenMP info
#include <stdexcept>  // Para std::runtime_error
#include <algorithm>  // Para std::replace

// Inclui o header com a lógica de comparação e funções auxiliares/solvers
#include "solver_comparison.hpp"
#include "include/types.hpp"
#include "include/config.hpp"

// Removida a função create_output_directories daqui pois já existe em solver_comparison.hpp

// --- Função Principal (SIMPLIFICADA) ---
int main(int argc, char *argv[]) {
    std::cout << "======================================================" << std::endl;
    std::cout << " Comparativo: CGNR Standard vs Pre-condicionado vs FISTA" << std::endl;
    std::cout << "======================================================" << std::endl;

    // Carrega a configuração do YAML
    Config config = load_config("config.yaml");
    if (config.run_pipelines.empty()) {
        std::cerr << "[ERRO] Nenhum pipeline definido no arquivo de configuração." << std::endl;
        return 1;
    }

    // Pega apenas o primeiro pipeline (Sparse_Comparison)
    const auto &pipeline = config.run_pipelines[0];

    // Prepara o vetor de testes baseado no pipeline
    std::cout << "\n[INFO] Pipeline selecionado: " << pipeline.name << std::endl;
    std::cout << "[INFO] Datasets configurados: ";
    for (const auto &name: pipeline.dataset_names) {
        std::cout << name << " ";
    }
    std::cout << std::endl;

    std::vector<DatasetConfig> tests;
    for (const auto &dataset_name: pipeline.dataset_names) {
        std::cout << "[INFO] Processando dataset: " << dataset_name << std::endl;
        auto it = config.dataset_map.find(dataset_name);
        if (it != config.dataset_map.end()) {
            tests.push_back(*it->second);
            std::cout << "[INFO] Dataset " << dataset_name << " adicionado para processamento" << std::endl;
        } else {
            std::cout << "[AVISO] Dataset " << dataset_name << " não encontrado no mapa de configurações" << std::endl;
        }
    }

    std::cout << "[INFO] Total de datasets a serem processados: " << tests.size() << std::endl;

    // Configuração OpenMP
    Eigen::setNbThreads(omp_get_max_threads());
    std::cout << "\n[INFO] Usando " << Eigen::nbThreads() << " threads para os calculos Eigen.\n" << std::endl;

    // --- Pré-processamento (Apenas garante que o .sparse.bin existe) ---
    std::cout << "[INFO] Verificando/Criando arquivos binarios esparsos..." << std::endl;
    for (const auto &config: tests) {
        std::filesystem::path h_path = config.h_matrix_csv; // Correção aqui: usa h_matrix_csv
        if (!std::filesystem::exists(h_path)) {
            std::cerr << "[ERRO] Arquivo CSV da matriz H nao encontrado: " << config.h_matrix_csv << std::endl;
            return 1;
        }
        std::filesystem::path data_dir = h_path.parent_path();
        std::filesystem::path sparse_bin_fs_path = data_dir / (h_path.filename().string() + ".sparse.bin");
        std::string sparse_bin_path = sparse_bin_fs_path.string();

        if (!std::filesystem::exists(sparse_bin_fs_path)) {
            std::cout << "[AVISO] Criando arquivo binario esparso para " << config.h_matrix_csv << " em " <<
                    sparse_bin_path << "..." << std::endl;
            try {
                saveSparseMatrix(convertCsvToSparse(config.h_matrix_csv, config.image_rows * config.image_cols),
                                 sparse_bin_path);
                std::cout << "[SUCESSO] Arquivo binario esparso criado: " << sparse_bin_path << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "[ERRO] Falha ao criar arquivo binario esparso: " << e.what() << std::endl;
                return 1;
            }
        }
    }
    std::cout << "[INFO] Pre-processamento concluido.\n" << std::endl;

    // Padronizar o caminho de saída para corresponder ao que o Python espera
    std::filesystem::path output_dir = "../output_csv";
    try {
        create_output_directories(output_dir); // Usa a função do solver_comparison.hpp
        std::cout << "[INFO] Estrutura de diretorios criada: " << std::filesystem::absolute(output_dir) << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "[ERRO] Nao foi possivel criar os diretorios de saida: " << e.what() << std::endl;
        return 1;
    }

    // --- Loop Principal de Testes ---
    std::vector<std::tuple<std::string, PerformanceMetrics, PerformanceMetrics, PerformanceMetrics> > all_results;

    // Verificação adicional para garantir que estamos processando apenas os datasets selecionados
    std::cout << "\n[INFO] Verificando datasets a serem processados:" << std::endl;
    for (const auto &config: tests) {
        std::cout << "  - Dataset: " << config.name << " (" << config.description << ")" << std::endl;
        bool is_in_pipeline = std::find(pipeline.dataset_names.begin(),
                                        pipeline.dataset_names.end(),
                                        config.name) != pipeline.dataset_names.end();
        if (!is_in_pipeline) {
            std::cout << "[AVISO] Dataset " << config.name << " não está no pipeline atual. Ignorando." << std::endl;
            continue;
        }

        std::cout << "\n========================================\nINICIANDO TESTE: " << config.description <<
                // Usa .description
                "\n========================================" << std::endl;

        try {
            auto [fista_metrics, other_metrics] = run_sparse_comparison_with_fista(config);
            all_results.push_back({config.description, other_metrics.first, other_metrics.second, fista_metrics});
            std::cout << "Teste " << config.description << " concluido com sucesso." << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[ERRO FATAL] Falha ao executar comparacao para o teste " << config.description << ": " << e.
                    what() << std::endl;
        }
    }

    // --- Tabela Final de Resultados ---
    std::cout <<
            "\n\n======================================================================================================================================================"
            << std::endl;
    std::cout << "                                                    RELATORIO COMPARATIVO FINAL" << std::endl;
    std::cout <<
            "======================================================================================================================================================"
            << std::endl;
    std::cout << std::left
            << std::setw(22) << "Teste"
            << std::setw(28) << "Metodo"
            << std::setw(15) << "RAM (MB)"
            << std::setw(20) << "T. Carga (ms)"
            << std::setw(20) << "T. Solver (ms)"
            << std::setw(12) << "Iteracoes"
            << std::setw(15) << "Erro Final"
            << std::setw(15) << "Epsilon Final"
            << std::setw(15) << "Convergiu"
            << std::setw(15) << "Speedup"
            << std::endl;
    std::cout <<
            "------------------------------------------------------------------------------------------------------------------------------------------------------"
            << std::endl;

    for (const auto &[test_name, std_metrics, precond_metrics, fista_metrics]: all_results) {
        auto print_metric = [&](const PerformanceMetrics &m, const std::string &method_name, bool is_baseline) {
            if (m.load_time_ms <= 0 && m.solve_time_ms <= 0) {
                std::cout << std::left
                        << std::setw(22) << (is_baseline ? test_name : "")
                        << std::setw(28) << method_name
                        << std::setw(15 + 20 + 20 + 12 + 15 + 15 + 15) << " [FALHOU]"
                        << std::setw(14) << std::right << (is_baseline ? "BASELINE" : "N/A")
                        << std::endl;
                return;
            }

            std::cout << std::left
                    << std::setw(22) << (is_baseline ? test_name : "")
                    << std::setw(28) << method_name
                    << std::fixed << std::setprecision(2) << std::setw(15) << m.estimated_ram_mb
                    << std::fixed << std::setprecision(2) << std::setw(20) << m.load_time_ms
                    << std::fixed << std::setprecision(2) << std::setw(20) << m.solve_time_ms
                    << std::setw(12) << m.iterations
                    << std::scientific << std::setprecision(3) << std::setw(14) << m.final_error
                    << std::scientific << std::setprecision(3) << std::setw(14) << m.final_epsilon
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
        print_metric(fista_metrics, "FISTA L1", false);

        std::cout <<
                "------------------------------------------------------------------------------------------------------------------------------------------------------"
                << std::endl;
    }

    std::cout << "\nBenchmark concluido." << std::endl;
    std::cout << "[INFO] Para visualizar os graficos de convergencia, execute:" << std::endl;
    std::cout << "[INFO] python ../data/plot_convergence.py" << std::endl;
    return 0;
}
