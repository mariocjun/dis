/**
 * C++ Ultrasound Reconstruction HTTP Server (Optimized with Caching)
 */

// Note: Windows headers removed due to SDK compatibility issues with PSAPI
// C++ RAM metrics disabled - Python server reports RAM correctly for benchmark

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "httplib.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "solver_comparison.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ---------- Global State with Caching ----------
struct ServerState {
  std::mutex mtx; // Protects shared resources if needed

  // Job counters (Atomic for thread safety)
  std::atomic<int64_t> jobs_total{0};
  std::atomic<int64_t> jobs_completed{0};
  std::atomic<int64_t> jobs_failed{0};
  std::atomic<int64_t> jobs_in_progress{0};
  std::atomic<int64_t> queue_length{0};

  std::chrono::steady_clock::time_point start_time;
  int num_threads = 4;

  struct DatasetInfo {
    std::string name;
    std::string h_matrix_csv;
    std::string g_signal_csv;
    int image_rows;
    int image_cols;
  };
  std::map<std::string, DatasetInfo> datasets;

  // --- MATRIX CACHE ---
  std::mutex cache_mtx;
  std::map<std::string, Eigen::SparseMatrix<double>> matrix_cache;
  // --------------------

  double epsilon_tolerance = 1e-4;
  int max_iterations = 10;
  fs::path output_dir;
  fs::path config_dir;

  void load_config(const std::string &config_path) {
    YAML::Node config = YAML::LoadFile(config_path);
    config_dir = fs::path(config_path).parent_path();

    if (config["settings"]) {
      YAML::Node settings = config["settings"];
      epsilon_tolerance = settings["epsilon_tolerance"].as<double>(1e-4);
      max_iterations = settings["max_iterations"].as<int>(10);
      num_threads = settings["num_omp_threads"].as<int>(4);
      std::string out_dir =
          settings["output_base_dir"].as<std::string>("execs/cpp_server");
      output_dir = fs::path(out_dir);
    }

    if (config["datasets"]) {
      for (const YAML::Node &ds : config["datasets"]) {
        DatasetInfo info;
        info.name = ds["name"].as<std::string>();
        info.h_matrix_csv = ds["h_matrix_csv"].as<std::string>();
        info.g_signal_csv = ds["g_signal_csv"].as<std::string>();
        info.image_rows = ds["image_rows"].as<int>();
        info.image_cols = ds["image_cols"].as<int>();
        datasets[info.name] = info;
      }
    }

    fs::create_directories(output_dir / "images");
    fs::create_directories(output_dir / "telemetry");

#ifdef _OPENMP
    if (num_threads > 0)
      omp_set_num_threads(num_threads);
#endif
    Eigen::setNbThreads(num_threads > 0 ? num_threads : 4);
  }

  // Helper to get or load matrix safely
  Eigen::SparseMatrix<double> get_matrix(const std::string &dataset_id,
                                         const fs::path &h_path, int rows,
                                         int cols) {
    std::lock_guard<std::mutex> lock(cache_mtx);

    // Check cache first
    auto it = matrix_cache.find(dataset_id);
    if (it != matrix_cache.end()) {
      return it->second; // Return copy (SparseMatrix copy is relatively cheap
                         // compared to I/O)
    }

    // Load if not in cache
    std::cout << "[CACHE] Loading matrix for " << dataset_id
              << " into memory..." << std::endl;
    Eigen::SparseMatrix<double> H;

    fs::path sparse_bin_path =
        h_path.parent_path() / (h_path.filename().string() + ".sparse.bin");

    if (fs::exists(sparse_bin_path)) {
      H = loadSparseMatrix(sparse_bin_path.string());
    } else {
      std::cout << "[CACHE] Converting CSV to sparse (One-time cost)..."
                << std::endl;
      H = convertCsvToSparse(h_path.string(), rows * cols);
      // Save binary for next run (optional, but good practice)
      try {
        saveSparseMatrix(H, sparse_bin_path.string());
      } catch (...) {
      }
    }

    matrix_cache[dataset_id] = H;
    return H;
  }
};

ServerState g_state;

// ---------- Utility Functions ----------

std::string get_iso_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%S");
  ss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
  return ss.str();
}

// Get current process memory usage in MB
// Note: PSAPI removed due to Windows SDK compatibility - returns 0
// Python server reports RAM metrics correctly for the benchmark
double get_current_memory_mb() { return 0.0; }

void save_job_metrics(const json &metrics) {
  // Use a mutex for file writing to avoid race conditions
  static std::mutex file_mtx;
  std::lock_guard<std::mutex> lock(file_mtx);

  fs::path csv_path = g_state.output_dir / "telemetry" / "job_metrics.csv";
  bool write_header = !fs::exists(csv_path);

  std::ofstream file(csv_path, std::ios::app);
  if (write_header) {
    file << "job_id,timestamp_start,timestamp_end,server,dataset_id,gain,seed,"
         << "iterations,final_error,final_epsilon,converged,latency_ms,solver_"
            "time_ms,"
         << "ram_peak_mb,cpu_avg_pct\n";
  }

  file << metrics["job_id"].get<std::string>() << ","
       << metrics["timestamp_start"].get<std::string>() << ","
       << metrics["timestamp_end"].get<std::string>() << ","
       << "cpp," << metrics["dataset_id"].get<std::string>() << ","
       << metrics["gain"].get<double>() << "," << metrics.value("seed", 0)
       << "," << metrics["iterations"].get<int>() << "," << std::scientific
       << metrics["final_error"].get<double>() << "," << std::scientific
       << metrics["final_epsilon"].get<double>() << ","
       << (metrics["converged"].get<bool>() ? "true" : "false") << ","
       << std::fixed << metrics["latency_ms"].get<double>() << "," << std::fixed
       << metrics["solver_time_ms"].get<double>() << "," << std::fixed
       << metrics.value("ram_peak_mb", 0.0) << "," << std::fixed
       << metrics.value("cpu_avg_pct", 0.0) << "\n";
}

void save_image_csv(const Eigen::VectorXd &image, const fs::path &path,
                    int rows, int cols) {
  std::ofstream file(path);
  file << std::scientific << std::setprecision(10);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (j > 0)
        file << ",";
      file << image(i * cols + j);
    }
    file << "\n";
  }
}

void save_metadata_json(const json &metadata, const fs::path &path) {
  std::ofstream file(path);
  file << metadata.dump(2);
}

fs::path resolve_path(const fs::path &base_dir, const std::string &path_str) {
  fs::path p(path_str);
  if (p.is_absolute())
    return p;
  return base_dir / p;
}

// ---------- HTTP Handlers ----------

void handle_health(const httplib::Request &req, httplib::Response &res) {
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - g_state.start_time)
                    .count();
  json response = {
      {"status", "healthy"}, {"server", "cpp"}, {"uptime_seconds", uptime}};
  res.set_content(response.dump(), "application/json");
}

void handle_metrics(const httplib::Request &req, httplib::Response &res) {
  std::stringstream ss;
  ss << "# HELP jobs_total Total jobs\n# TYPE jobs_total counter\njobs_total "
     << g_state.jobs_total.load() << "\n"
     << "# HELP jobs_completed Jobs completed\n# TYPE jobs_completed "
        "counter\njobs_completed "
     << g_state.jobs_completed.load() << "\n"
     << "# HELP jobs_failed Jobs failed\n# TYPE jobs_failed "
        "counter\njobs_failed "
     << g_state.jobs_failed.load() << "\n";
  res.set_content(ss.str(), "text/plain");
}

void handle_solve(const httplib::Request &req, httplib::Response &res) {
  std::string timestamp_start = get_iso_timestamp();
  auto start_time = std::chrono::high_resolution_clock::now();

  g_state.jobs_total++;
  g_state.jobs_in_progress++;

  try {
    json request_data = json::parse(req.body);
    std::string job_id = request_data.value(
        "job_id", "cpp_job_" + std::to_string(g_state.jobs_total.load()));
    std::string dataset_id = request_data.value("dataset_id", "");
    double gain = request_data.value("gain", 1.0);
    int seed = request_data.value("seed", 0);
    double epsilon_tolerance =
        request_data.value("epsilon_tolerance", g_state.epsilon_tolerance);
    int max_iterations =
        request_data.value("max_iterations", g_state.max_iterations);

    std::cout << "[JOB " << job_id << "] Start: " << dataset_id << std::endl;

    auto it = g_state.datasets.find(dataset_id);
    if (it == g_state.datasets.end()) {
      throw std::runtime_error("Unknown dataset_id: " + dataset_id);
    }
    const auto &ds = it->second;

    fs::path h_path = resolve_path(g_state.config_dir, ds.h_matrix_csv);
    fs::path g_path = resolve_path(g_state.config_dir, ds.g_signal_csv);

    // Load G (Fast, no caching needed usually)
    Eigen::VectorXd g_signal = loadVectorData(g_path.string());

    // Load H (Cached)
    Eigen::SparseMatrix<double> H_matrix =
        g_state.get_matrix(dataset_id, h_path, ds.image_rows, ds.image_cols);

    // Apply gain
    g_signal *= gain;

    // Normalize (Important: H is shared in cache, so we must NOT modify H in
    // place if it affects others. However, normalize_system_rows modifies H.
    // FIX: Since H is cached, we should normalize it ONCE when loading into
    // cache. BUT, g_signal changes every request (gain). Strategy: We will make
    // a COPY of H for the solver OR assume H is already normalized in cache?
    // Normalization depends on H rows. H rows don't change.
    // So we can normalize H once in cache. But we need to normalize g using H's
    // row norms. For safety/simplicity in this fix: We will copy H. Copying a
    // sparse matrix is fast enough compared to loading from disk.
    Eigen::SparseMatrix<double> H_copy = H_matrix;

    normalize_system_rows(H_copy, g_signal);

    // Solver - Enable intermediate image saving for carousel
    auto t0 = std::chrono::high_resolution_clock::now();
    ReconstructionResult result = run_cgnr_solver_epsilon_save_iters(
        g_signal, H_copy, epsilon_tolerance, max_iterations,
        "cpp_" + job_id, // Prefix for iteration files
        g_state.output_dir.string(), ds.image_rows, ds.image_cols,
        true); // Save intermediate images for carousel
    auto t1 = std::chrono::high_resolution_clock::now();
    double solver_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ram_peak_mb = get_current_memory_mb();

    std::string timestamp_end = get_iso_timestamp();
    auto end_time = std::chrono::high_resolution_clock::now();
    double latency_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time)
            .count();

    // Save outputs - Add cpp_ prefix to prevent Python from overwriting
    fs::path img_csv_path =
        g_state.output_dir / "images" / ("cpp_" + job_id + "_image.csv");
    save_image_csv(result.image, img_csv_path, ds.image_rows, ds.image_cols);

    json metadata = {
        {"job_id", job_id},
        {"algorithm", "CGNR"},
        {"dataset_id", dataset_id},
        {"image_size", {{"rows", ds.image_rows}, {"cols", ds.image_cols}}},
        {"gain", gain},
        {"seed", seed},
        {"iterations", result.iterations},
        {"final_error", result.final_error},
        {"final_epsilon", result.final_epsilon},
        {"converged", result.converged},
        {"timestamp_start", timestamp_start},
        {"timestamp_end", timestamp_end},
        {"solver_time_ms", result.execution_time_ms},
        {"latency_ms", latency_ms},
        {"server", "cpp"},
        {"ram_peak_mb", ram_peak_mb},
        {"cpu_avg_pct", 0.0},
        {"image_csv_path", "images/cpp_" + job_id + "_image.csv"},
        {"metadata_json_path", "images/cpp_" + job_id + "_meta.json"}};

    fs::path meta_json_path =
        g_state.output_dir / "images" / ("cpp_" + job_id + "_meta.json");
    save_metadata_json(metadata, meta_json_path);
    save_job_metrics(metadata);

    g_state.jobs_completed++;
    g_state.jobs_in_progress--;

    json response = {{"job_id", job_id}, {"status", "completed"}};
    response.update(metadata);
    res.set_content(response.dump(), "application/json");

    std::cout << "[JOB " << job_id << "] Success. Solver: " << solver_ms << "ms"
              << std::endl;

  } catch (const std::exception &e) {
    g_state.jobs_failed++;
    g_state.jobs_in_progress--;
    std::cerr << "[ERROR] Job failed: " << e.what() << std::endl;
    json error_response = {{"error", e.what()}, {"status", "failed"}};
    res.status = 500;
    res.set_content(error_response.dump(), "application/json");
  } catch (...) {
    g_state.jobs_failed++;
    g_state.jobs_in_progress--;
    std::cerr << "[ERROR] Job failed: Unknown fatal error" << std::endl;
    res.status = 500;
    res.set_content("{\"error\": \"Unknown fatal error\"}", "application/json");
  }
}

int main(int argc, char *argv[]) {
  std::string config_path = "config.yaml";
  int port = 5002;
  std::string host = "0.0.0.0";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      config_path = argv[++i];
    else if (arg == "--port" && i + 1 < argc)
      port = std::stoi(argv[++i]);
  }

  fs::path cfg_path(config_path);
  if (!fs::exists(cfg_path))
    cfg_path = fs::path(argv[0]).parent_path() / "config.yaml";

  if (fs::exists(cfg_path)) {
    g_state.load_config(cfg_path.string());
    std::cout << "[INFO] Config loaded. Datasets: " << g_state.datasets.size()
              << std::endl;
  } else {
    std::cerr << "[WARN] Config not found: " << config_path << std::endl;
  }

  g_state.start_time = std::chrono::steady_clock::now();
  httplib::Server svr;
  svr.Get("/health", handle_health);
  svr.Get("/metrics", handle_metrics);
  svr.Post("/solve", handle_solve);

  std::cout << "[INFO] C++ Server listening on " << host << ":" << port
            << std::endl;
  svr.listen(host.c_str(), port);
  return 0;
}