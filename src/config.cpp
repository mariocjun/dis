#include "../include/config.hpp"
#include <iostream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>


Config load_config(const std::string &config_path) {
  Config config;
  try {
    YAML::Node yaml = YAML::LoadFile(config_path);

    // Carrega as configurações globais
    if (yaml["settings"]) {
      auto settings = yaml["settings"];
      config.settings.output_base_dir =
          settings["output_base_dir"].as<std::string>();
      config.settings.epsilon_tolerance =
          settings["epsilon_tolerance"].as<double>();
      config.settings.max_iterations = settings["max_iterations"].as<int>();
      if (settings["l_curve_iterations"]) {
        config.settings.l_curve_iterations =
            settings["l_curve_iterations"].as<int>();
      }
      config.settings.save_intermediate_images =
          settings["save_intermediate_images"].as<bool>();
      config.settings.num_omp_threads = settings["num_omp_threads"].as<int>();
    }

    // Carrega os datasets
    if (yaml["datasets"]) {
      for (const auto &dataset : yaml["datasets"]) {
        DatasetConfig dc;
        dc.name = dataset["name"].as<std::string>();
        dc.description = dataset["description"].as<std::string>();
        dc.h_matrix_csv = dataset["h_matrix_csv"].as<std::string>();
        dc.g_signal_csv = dataset["g_signal_csv"].as<std::string>();
        dc.image_rows = dataset["image_rows"].as<int>();
        dc.image_cols = dataset["image_cols"].as<int>();
        config.datasets.push_back(dc);
      }
    }

    // Carrega os métodos
    if (yaml["methods"]) {
      for (const auto &method : yaml["methods"]) {
        MethodConfig mc;
        mc.name = method["name"].as<std::string>();
        mc.description = method["description"].as<std::string>();
        mc.solver = method["solver"].as<std::string>();
        mc.use_binary = method["use_binary"].as<bool>();
        mc.is_baseline = method["is_baseline"].as<bool>();
        config.methods.push_back(mc);
      }
    }

    // Carrega os pipelines
    if (yaml["run_pipelines"]) {
      for (const auto &pipeline : yaml["run_pipelines"]) {
        PipelineConfig pc;
        pc.name = pipeline["name"].as<std::string>();
        pc.description = pipeline["description"].as<std::string>();
        pc.method_names = pipeline["methods"].as<std::vector<std::string>>();
        pc.dataset_names = pipeline["datasets"].as<std::vector<std::string>>();
        config.run_pipelines.push_back(pc);
      }
    }

    // Popula os mapas para acesso rápido
    config.populate_maps();

    // Validação do pipeline
    std::cout << "\n[INFO] Validando pipeline..." << std::endl;
    for (const auto &pipeline : config.run_pipelines) {
      std::cout << "\nPipeline: " << pipeline.name << std::endl;
      std::cout << "Datasets configurados:" << std::endl;
      for (const auto &dataset_name : pipeline.dataset_names) {
        if (config.dataset_map.find(dataset_name) == config.dataset_map.end()) {
          throw std::runtime_error(
              "Dataset '" + dataset_name + "' configurado no pipeline '" +
              pipeline.name +
              "' não foi encontrado na lista de datasets disponíveis");
        }
        std::cout << "  - " << dataset_name << std::endl;
      }

      std::cout << "Métodos configurados:" << std::endl;
      for (const auto &method_name : pipeline.method_names) {
        if (config.method_map.find(method_name) == config.method_map.end()) {
          throw std::runtime_error(
              "Método '" + method_name + "' configurado no pipeline '" +
              pipeline.name +
              "' não foi encontrado na lista de métodos disponíveis");
        }
        std::cout << "  - " << method_name << std::endl;
      }
    }

    std::cout << "\n[INFO] Configuração carregada com sucesso de: "
              << config_path << std::endl;
    if (!config.run_pipelines.empty()) {
      std::cout << "[INFO] Pipeline ativo: " << config.run_pipelines[0].name
                << std::endl;
      std::cout << "[INFO] Dataset(s) a processar:";
      for (const auto &dataset : config.run_pipelines[0].dataset_names) {
        std::cout << " " << dataset;
      }
      std::cout << std::endl;
    }

  } catch (const YAML::Exception &e) {
    throw std::runtime_error("Erro ao carregar configuração YAML: " +
                             std::string(e.what()));
  }

  return config;
}
