#include "../include/benchmark_runner.hpp" // Inclui a declaração da função principal do benchmark
#include "../include/types.hpp"           // Inclui as definições das structs (Resultados, Métricas, etc.)
#include "../include/config.hpp"          // Inclui a definição da classe Config
#include "../include/io_utils.hpp"        // Inclui as declarações das funções de I/O (load*, save*)
#include "../include/solvers.hpp"         // Inclui as declarações dos solvers (CGNR, Precondicionado, Fixo)
#include "../include/utils.hpp"           // Inclui a declaração da função de normalização

#include <iostream>                      // Para std::cout, std::cerr
#include <string>                        // Para std::string
#include <vector>                        // Para std::vector
#include <map>                           // Para std::map (armazenar resultados)
#include <chrono>                        // Para medição de tempo (high_resolution_clock)
#include <filesystem>                    // Para manipulação de caminhos (C++17)
#include <stdexcept>                     // Para std::runtime_error
#include <Eigen/Sparse>                  // Para Eigen::SparseMatrix
#include <Eigen/Dense>                   // Para Eigen::MatrixXd (caso reative testes densos)
#include <algorithm>                     // Para std::replace

// --- Implementação da Função Principal do Benchmark ---

BenchmarkResults run_benchmarks(const Config& config) {
    BenchmarkResults benchmark_results; // Estrutura para guardar todos os resultados
    const GlobalSettings& settings = config.settings; // Atalho para configurações globais

    std::cout << "\n======================================================" << std::endl;
    std::cout << "               INICIANDO BENCHMARKS" << std::endl;
    std::cout << "======================================================" << std::endl;

    // Garante que o diretório base de saída exista e cria subdiretório para CSVs
    std::filesystem::path base_output_dir = settings.output_base_dir;
    std::filesystem::path results_csv_dir = base_output_dir / "results_csv";
    try {
        std::filesystem::create_directories(results_csv_dir); // Cria o subdiretório para CSVs
        std::cout << "[INFO] Diretorio de saida CSV verificado/criado: " << std::filesystem::absolute(results_csv_dir) << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        // Lança um erro se não conseguir criar o diretório essencial
        throw std::runtime_error("Nao foi possivel criar o diretorio de saida CSV: " + results_csv_dir.string() + " - " + e.what());
    }
    // Poderia criar subpastas para logs e PNGs aqui também, se necessário, de forma similar

    // Itera sobre cada pipeline definido no config.yaml
    for (const auto& pipeline : config.run_pipelines) {
        std::cout << "\n--- Iniciando Pipeline: " << pipeline.name << " (" << pipeline.description << ") ---" << std::endl;

        // Itera sobre cada dataset neste pipeline
        for (const std::string& dataset_name : pipeline.dataset_names) {
            // Encontra a configuração do dataset pelo nome
            auto ds_it = config.dataset_map.find(dataset_name);
            if (ds_it == config.dataset_map.end()) {
                std::cerr << "[AVISO] Dataset '" << dataset_name << "' definido no pipeline '" << pipeline.name << "' nao encontrado na lista de datasets. Pulando." << std::endl;
                continue;
            }
            const DatasetConfig& current_dataset = *ds_it->second;

            std::cout << "\n  -- Processando Dataset: " << current_dataset.description << " --" << std::endl;

            // Garante que a entrada no mapa de resultados para este dataset exista
            benchmark_results.results[current_dataset.name];

            // Itera sobre cada método neste pipeline
            for (const std::string& method_name : pipeline.method_names) {
                // Encontra a configuração do método pelo nome
                auto m_it = config.method_map.find(method_name);
                if (m_it == config.method_map.end()) {
                    std::cerr << "[AVISO] Metodo '" << method_name << "' definido no pipeline '" << pipeline.name << "' nao encontrado na lista de metodos. Pulando." << std::endl;
                    continue;
                }
                const MethodConfig& current_method = *m_it->second;

                std::cout << "\n    - Aplicando Metodo: " << current_method.description << " -" << std::endl;

                // --- Variáveis para esta execução específica ---
                PerformanceMetrics current_metrics;
                ReconstructionResult solver_result;
                // **** CORREÇÃO: Declarações movidas para dentro do loop ****
                std::chrono::high_resolution_clock::time_point load_start, load_end;
                double load_time_ms = 0;
                // **** FIM DA CORREÇÃO ****
                double ram_mb = 0;

                try {
                    // --- Carregamento dos Dados ---
                    // **** CORREÇÃO: Usa h_csv_path consistentemente ****
                    std::filesystem::path h_csv_path = current_dataset.h_matrix_csv;
                    std::filesystem::path data_dir = h_csv_path.parent_path();
                    // **** FIM DA CORREÇÃO ****
                    std::filesystem::path h_load_path; // Caminho final (CSV ou BIN) a ser carregado

                    bool is_sparse = (current_method.solver.find("cgnr") != std::string::npos);

                    if (current_method.use_binary) {
                        std::string bin_suffix = is_sparse ? ".sparse.bin" : ".dense.bin";
                        h_load_path = data_dir / (h_csv_path.filename().string() + bin_suffix); // Usa h_csv_path
                        if (!std::filesystem::exists(h_load_path)) {
                            throw std::runtime_error("Arquivo binario necessario '" + h_load_path.string() + "' nao encontrado (execute o pre-processamento).");
                        }
                         std::cout << "[INFO] Carregando H de: " << h_load_path.string() << std::endl;
                    } else {
                        h_load_path = h_csv_path; // Usa h_csv_path
                         if (!std::filesystem::exists(h_load_path)) {
                             throw std::runtime_error("Arquivo CSV necessario '" + h_load_path.string() + "' nao encontrado.");
                         }
                         std::cout << "[INFO] Carregando H de: " << h_load_path.string() << std::endl;
                    }

                    // Inicia a contagem do tempo de carregamento
                    load_start = std::chrono::high_resolution_clock::now();
                    // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
                    Eigen::VectorXd g = loadVectorData(current_dataset.g_signal_csv);
                    // **** FIM DA CORREÇÃO ****

                    // Carrega H esparso ou denso
                    if (is_sparse) {
                        Eigen::SparseMatrix<double> H_sparse;
                        if (current_method.use_binary) {
                            H_sparse = loadSparseMatrix(h_load_path.string());
                        } else {
                             std::cout << "[INFO] Convertendo CSV esparso para memoria..." << std::endl;
                            H_sparse = convertCsvToSparse(h_load_path.string(), current_dataset.image_rows * current_dataset.image_cols);
                        }
                        // Finaliza a contagem do tempo de carregamento
                        load_end = std::chrono::high_resolution_clock::now();
                        // Calcula RAM estimada
                        ram_mb = static_cast<double>(H_sparse.nonZeros() * (sizeof(double) + sizeof(int)) + (H_sparse.outerSize() + 1) * sizeof(int)) / (1024.0 * 1024.0);

                        // Normalização
                        normalize_system_rows(H_sparse, g);

                        // Debug z0
                        Eigen::VectorXd z0 = H_sparse.transpose() * g;
                        std::cout << "[DEBUG " << current_method.name << "] Norma de z0 (H^T * g norm): " << z0.norm() << std::endl;

                        // Seleciona e Chama o Solver Esparso
                        std::string filename_prefix = "image_" + current_dataset.name + "_" + current_method.name;

                        if (current_method.solver == "cgnr_standard") {
                            std::cout << "[INFO] Chamando solver: cgnr_standard..." << std::endl;
                            solver_result = run_cgnr_solver_epsilon_save_iters(g, H_sparse, settings.epsilon_tolerance, settings.max_iterations,
                                (settings.save_intermediate_images ? filename_prefix : ""), results_csv_dir, current_dataset.image_rows, current_dataset.image_cols);
                        } else if (current_method.solver == "cgnr_preconditioned") {
                             std::cout << "[INFO] Chamando solver: cgnr_preconditioned..." << std::endl;
                             solver_result = run_cgnr_solver_preconditioned_save_iters(g, H_sparse, settings.epsilon_tolerance, settings.max_iterations,
                                (settings.save_intermediate_images ? filename_prefix : ""), results_csv_dir, current_dataset.image_rows, current_dataset.image_cols);
                        }
                        // else if (current_method.solver == "fista") { ... }
                        else {
                            throw std::runtime_error("Solver esparso desconhecido ou nao implementado: " + current_method.solver);
                        }

                         // Gerar CSVs de histórico/L-curve
                         Eigen::SparseMatrix<double> H_fixed;
                         if (current_method.use_binary) H_fixed = loadSparseMatrix(h_load_path.string());
                         else H_fixed = convertCsvToSparse(h_load_path.string(), current_dataset.image_rows * current_dataset.image_cols);
                         // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
                         Eigen::VectorXd g_fixed = loadVectorData(current_dataset.g_signal_csv);
                         // **** FIM DA CORREÇÃO ****
                         normalize_system_rows(H_fixed, g_fixed);
                         ReconstructionResult res_fixed = run_cgnr_solver_fixed_iter(g_fixed, H_fixed, settings.max_iterations);
                         std::filesystem::path hist_path = results_csv_dir / ("convergence_history_" + current_dataset.name + "_" + current_method.name + ".csv");
                         std::filesystem::path lcurve_path = results_csv_dir / ("lcurve_" + current_dataset.name + "_" + current_method.name + ".csv");
                         saveHistoryToCSV(res_fixed.residual_history, hist_path.string());
                         saveLcurveToCSV(res_fixed, lcurve_path.string());

                    } else { // Carrega Denso
                        Eigen::MatrixXd H_dense;
                        if (current_method.use_binary) {
                            H_dense = loadDenseMatrix(h_load_path.string());
                        } else {
                            std::cout << "[INFO] Convertendo CSV denso para memoria..." << std::endl;
                            H_dense = loadDenseData(h_load_path.string());
                        }
                        load_end = std::chrono::high_resolution_clock::now();
                        ram_mb = static_cast<double>(H_dense.rows()) * H_dense.cols() * sizeof(double) / (1024.0 * 1024.0);

                        normalize_system_rows(H_dense, g);
                        Eigen::VectorXd z0 = H_dense.transpose() * g;
                        std::cout << "[DEBUG " << current_method.name << "] Norma de z0 (H^T * g norm): " << z0.norm() << std::endl;

                        std::string filename_prefix = "image_" + current_dataset.name + "_" + current_method.name;
                        if (current_method.solver == "cgnr_standard") {
                             std::cout << "[INFO] Chamando solver: cgnr_standard (Denso)..." << std::endl;
                             solver_result = run_cgnr_solver_epsilon_save_iters(g, H_dense, settings.epsilon_tolerance, settings.max_iterations,
                                (settings.save_intermediate_images ? filename_prefix : ""), results_csv_dir, current_dataset.image_rows, current_dataset.image_cols);
                        }
                        else {
                             throw std::runtime_error("Solver denso desconhecido ou nao suportado: " + current_method.solver);
                        }

                         // Gerar CSVs de histórico/L-curve
                         Eigen::MatrixXd H_fixed;
                         if (current_method.use_binary) H_fixed = loadDenseMatrix(h_load_path.string());
                         else H_fixed = loadDenseData(h_load_path.string());
                         // **** CORREÇÃO: g_signal_path -> g_signal_csv ****
                         Eigen::VectorXd g_fixed = loadVectorData(current_dataset.g_signal_csv);
                         // **** FIM DA CORREÇÃO ****
                         normalize_system_rows(H_fixed, g_fixed);
                         ReconstructionResult res_fixed = run_cgnr_solver_fixed_iter(g_fixed, H_fixed, settings.max_iterations);
                         std::filesystem::path hist_path = results_csv_dir / ("convergence_history_" + current_dataset.name + "_" + current_method.name + ".csv");
                         std::filesystem::path lcurve_path = results_csv_dir / ("lcurve_" + current_dataset.name + "_" + current_method.name + ".csv");
                         saveHistoryToCSV(res_fixed.residual_history, hist_path.string());
                         saveLcurveToCSV(res_fixed, lcurve_path.string());
                    }

                    // --- Coleta Métricas ---
                    // **** CORREÇÃO: Linha que dava erro de 'start_load' ****
                    load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
                    // **** FIM DA CORREÇÃO ****
                    current_metrics.load_time_ms = load_time_ms;
                    current_metrics.estimated_ram_mb = ram_mb;
                    current_metrics.solve_time_ms = solver_result.execution_time_ms;
                    current_metrics.iterations = solver_result.iterations;
                    current_metrics.final_error = solver_result.final_error;
                    current_metrics.final_epsilon = solver_result.final_epsilon;
                    current_metrics.converged = solver_result.converged;
                    current_metrics.optimization_type = current_method.solver; // Guarda qual solver foi usado


                    // Salva a imagem final (se não salvou intermediárias)
                    if (!settings.save_intermediate_images && solver_result.image.size() > 0) {
                         std::string final_img_filename = "image_" + current_dataset.name + "_" + current_method.name + "_final.csv";
                         std::filesystem::path final_img_path = results_csv_dir / final_img_filename;
                         try {
                              saveImageVectorToCsv(solver_result.image, final_img_path.string(), current_dataset.image_rows, current_dataset.image_cols);
                         } catch (const std::exception& e) {
                             std::cerr << "[AVISO] Falha ao salvar imagem final: " << e.what() << std::endl;
                         }
                    }


                } catch (const std::exception& e) {
                    std::cerr << "[ERRO] Falha ao processar Metodo '" << current_method.name << "' no Dataset '" << current_dataset.name << "': " << e.what() << std::endl;
                    current_metrics = PerformanceMetrics(); // Zera métricas
                    current_metrics.optimization_type = current_method.solver; // Atribui tipo mesmo em erro
                }

                // Armazena as métricas no mapa de resultados
                benchmark_results.results[current_dataset.name][current_method.name] = current_metrics;

                 if (current_method.is_baseline && benchmark_results.baseline_method == nullptr) {
                    benchmark_results.baseline_method = &current_method;
                 }

                std::cout << "    - Metodo '" << current_method.name << "' concluido." << std::endl;

            } // Fim loop methods
             std::cout << "  -- Dataset '" << current_dataset.name << "' concluido. --" << std::endl;
        } // Fim loop datasets
         std::cout << "--- Pipeline '" << pipeline.name << "' concluido. ---" << std::endl;
    } // Fim loop pipelines

    std::cout << "\n======================================================" << std::endl;
    std::cout << "               BENCHMARKS CONCLUIDOS" << std::endl;
    std::cout << "======================================================" << std::endl;

    return benchmark_results;
}