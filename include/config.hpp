#ifndef ULTRASOUNDBENCHMARK_CONFIG_HPP
#define ULTRASOUNDBENCHMARK_CONFIG_HPP

#include <iostream>
#include "types.hpp" // Inclui as definições de structs (DatasetConfig, MethodConfig, etc.)
#include <string>
#include <vector>
#include <map>

// Classe para armazenar a configuração completa lida do YAML
class Config {
public:
    GlobalSettings settings; // Configurações globais
    std::vector<DatasetConfig> datasets; // Lista de datasets disponíveis
    std::vector<MethodConfig> methods; // Lista de métodos disponíveis
    std::vector<PipelineConfig> run_pipelines; // Lista de pipelines a executar

    // Mapas para acesso rápido por nome (preenchidos após carregar)
    std::map<std::string, const DatasetConfig*> dataset_map;
    std::map<std::string, const MethodConfig*> method_map;

    // Construtor padrão
    Config() = default;

    // Função para preencher os mapas após carregar os vetores
    void populate_maps() {
        dataset_map.clear();
        method_map.clear();

        std::cout << "\n[INFO] Populando mapas de configuração..." << std::endl;

        std::cout << "[INFO] Datasets disponíveis:" << std::endl;
        for (const auto& dataset : datasets) {
            dataset_map[dataset.name] = &dataset;
            std::cout << "  - " << dataset.name << std::endl;
        }

        std::cout << "[INFO] Métodos disponíveis:" << std::endl;
        for (const auto& method : methods) {
            method_map[method.name] = &method;
            std::cout << "  - " << method.name << std::endl;
        }

        std::cout << "[INFO] Pipelines configurados:" << std::endl;
        for (const auto& pipeline : run_pipelines) {
            std::cout << "  Pipeline: " << pipeline.name << std::endl;
            std::cout << "    Datasets: ";
            for (const auto& dataset : pipeline.dataset_names) {
                std::cout << dataset << " ";
            }
            std::cout << std::endl;
            std::cout << "    Métodos: ";
            for (const auto& method : pipeline.method_names) {
                std::cout << method << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Declaração da função que carregará a configuração do arquivo YAML
Config load_config(const std::string& config_path);


#endif //ULTRASOUNDBENCHMARK_CONFIG_HPP