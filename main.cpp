/**
 * Servidor C++ para Reconstrução de Imagem
 * Usa sockets raw para comparação justa com Python.
 */

#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
using SocketType = SOCKET;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define SOCKET_ERROR_VAL SOCKET_ERROR
#define CLOSE_SOCKET closesocket
#else
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketType = int;
#define INVALID_SOCKET_VAL -1
#define SOCKET_ERROR_VAL -1
#define CLOSE_SOCKET close
#endif

#include "include/io_utils.hpp"
#include "include/solvers.hpp"

using json = nlohmann::json;

// --- Variáveis Globais ---
Eigen::MatrixXd H_30;
Eigen::MatrixXd H_30_T;
Eigen::MatrixXd H_60;
Eigen::MatrixXd H_60_T;
// H_large em RowMajor para otimização
RowMajorMatrix H_large_rm;
RowMajorMatrix H_large_T_rm;
// H_large em ColMajor para teste MKL (Solver Padrão)
Eigen::MatrixXd H_large_mkl;
Eigen::MatrixXd H_large_T_mkl;
RowMajorMatrix H_parse_rm;
RowMajorMatrix H_parse_T_rm;

double C_factor = 0.0;

// --- Carrega matriz inteligente (binário ou CSV) ---
void load_matrix_smart(const std::string &csv_path, Eigen::MatrixXd &mat) {
  std::string bin_path = csv_path + ".dense.bin";

  // Tenta carregar binário primeiro
  if (std::filesystem::exists(bin_path)) {
    std::cout << "[INFO] Carregando binário: " << bin_path << std::endl;
    mat = loadDenseMatrix(bin_path);
    return;
  }

  // Carrega CSV e salva binário
  std::cout << "[INFO] Convertendo CSV: " << csv_path << std::endl;
  mat = loadDenseData(csv_path);
  saveDenseMatrix(mat, bin_path);
  std::cout << "[INFO] Binário salvo: " << bin_path << std::endl;
}

// --- Calcula fator de redução (norma espectral) ---
double calculate_reduction_factor(const Eigen::MatrixXd &H,
                                  int iterations = 10) {
  int N = H.cols();
  Eigen::VectorXd b = Eigen::VectorXd::Random(N);
  b.normalize();

  for (int i = 0; i < iterations; ++i) {
    Eigen::VectorXd y = H * b;
    Eigen::VectorXd z = H.transpose() * y;
    b = z.normalized();
  }

  Eigen::VectorXd y = H * b;
  return y.squaredNorm();
}

// --- Parse rápido do sinal ---
// --- Parse rápido do sinal (Zero-Copy) ---
std::vector<double> fast_parse_signal(const std::string &json_str) {
  std::vector<double> signal;

  // Encontrar início do array "signal_g"
  const char *str = json_str.data();
  const char *key = "\"signal_g\"";
  const char *found = std::strstr(str, key);

  if (!found)
    return signal;

  // Encontrar o '['
  const char *array_start = std::strchr(found, '[');
  if (!array_start)
    return signal;

  signal.reserve(3600);

  const char *p = array_start + 1;
  const char *end = json_str.data() + json_str.size();

  while (p < end && *p != ']') {
    // Skip spaces/delimiters
    while (p < end && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r')) {
      p++;
    }

    if (p == end || *p == ']')
      break;

    double val;
    auto [ptr, ec] = std::from_chars(p, end, val);

    if (ec == std::errc()) {
      signal.push_back(val);
      p = ptr;
    } else {
      // Fallback or error (skip char)
      p++;
    }
  }

  return signal;
}

// --- Solver CGNR Otimizado ---
struct SolverResult {
  Eigen::VectorXd image;
  int iterations;
  double execution_time_ms;
};

SolverResult fast_cgnr_solve(const Eigen::VectorXd &g, const Eigen::MatrixXd &H,
                             const Eigen::MatrixXd &H_T, int max_iters) {
  const auto t0 = std::chrono::high_resolution_clock::now();

  const int n = H.cols();
  Eigen::VectorXd f = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd r = g;
  Eigen::VectorXd z = H_T * r;
  Eigen::VectorXd p = z;
  Eigen::VectorXd w(H.rows());
  Eigen::VectorXd z_new(n);

  double z_sq = z.squaredNorm();
  double lambda = z.array().abs().maxCoeff() * 0.10;
  if (lambda < 1e-9)
    lambda = 1e-9;

  double prev_residual = r.norm();
  int final_iter = max_iters;

  for (int i = 0; i < max_iters; ++i) {
    w.noalias() = H * p;
    double w_sq = w.squaredNorm();
    double p_sq = p.squaredNorm();
    double denom = w_sq + lambda * p_sq;

    if (denom < 1e-15) {
      final_iter = i + 1;
      break;
    }

    double alpha = z_sq / denom;
    f.noalias() += alpha * p;
    r.noalias() -= alpha * w;

    double current_residual = r.norm();
    if (i > 0 && std::fabs(current_residual - prev_residual) < 1e-4) {
      final_iter = i + 1;
      break;
    }
    prev_residual = current_residual;

    z_new.noalias() = H_T * r - lambda * f;
    double z_new_sq = z_new.squaredNorm();
    double beta = (z_sq > 1e-15) ? (z_new_sq / z_sq) : 0.0;

    p *= beta;
    p.noalias() += z_new;
    z.swap(z_new);
    z_sq = z_new_sq;
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  return {f, final_iter,
          std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

// --- Processar cliente ---
void handle_client(SocketType clientSocket) {
  std::vector<char> buffer(1024 * 1024);
  std::string received_data;

  while (true) {
    int bytesReceived = recv(clientSocket, buffer.data(), buffer.size(), 0);
    if (bytesReceived > 0) {
      received_data.append(buffer.data(), bytesReceived);
      if (received_data.back() == '}')
        break;
    } else {
      return;
    }
  }

  auto start_time = std::chrono::system_clock::now();
  auto t0 = std::chrono::high_resolution_clock::now();

  try {
    // Parse model_size
    std::string model_size = "30x30";
    size_t model_pos = received_data.find("\"model_size\"");
    if (model_pos != std::string::npos) {
      size_t quote1 = received_data.find('\"', model_pos + 12);
      size_t quote2 = received_data.find('\"', quote1 + 1);
      model_size = received_data.substr(quote1 + 1, quote2 - quote1 - 1);
    }

    // Parse signal
    std::vector<double> signal_vec = fast_parse_signal(received_data);
    Eigen::VectorXd g =
        Eigen::Map<Eigen::VectorXd>(signal_vec.data(), signal_vec.size());

    auto parse_end = std::chrono::high_resolution_clock::now();
    double parse_time =
        std::chrono::duration<double, std::milli>(parse_end - t0).count();

    // Select matrix
    Eigen::MatrixXd *H = nullptr;
    Eigen::MatrixXd *H_T = nullptr;

    if (model_size == "large") {
      // TESTE MKL: Usar solver padrão (H*p) com matriz ColMajor
      // Se MKL estiver ativo, isso chamará dgemv.
      // Substituindo o solver paralelo manual.

      // Eigen::setNbThreads(0); // Deixar MKL/OpenMP decidir threads

      SolverResult result = fast_cgnr_solve(g, H_large_mkl, H_large_T_mkl, 10);

      // Output igual ao handle_client anterior
      std::cout << "[OK] Solver(MKL)=" << (int)result.execution_time_ms
                << "ms, Parse=" << (int)parse_time
                << "ms, Iters=" << result.iterations << "\n";

      // Serializa resposta
      std::ostringstream response_oss;
      response_oss << "{";
      response_oss << "\"image\": [";
      // Convert VectorXd to string loop... ou usar mesma logica
      // ...
      // Simplificação: copiar código de serialização ou refatorar?
      // Refatorar serialização seria melhor mas vou inline aqui para rapidez
      for (int i = 0; i < result.image.size(); ++i) {
        response_oss << std::fixed << std::setprecision(6) << result.image[i];
        if (i < result.image.size() - 1)
          response_oss << ",";
      }
      response_oss << "],";
      response_oss << "\"solver_time_ms\": " << result.execution_time_ms << ",";
      response_oss << "\"iterations\": " << result.iterations << ",";
      response_oss << "\"success\": true";
      response_oss << "}";

      std::string response = response_oss.str();
      send(clientSocket, response.c_str(), response.length(), 0);
      return; // Fim, pois já enviamos

    } else if (model_size == "30x30") {
      H = &H_30;
      H_T = &H_30_T;
      Eigen::setNbThreads(1);
    } else {
      H = &H_60;
      H_T = &H_60_T;
      Eigen::setNbThreads(1);
    }

    if (H->rows() == 0) {
      std::string err = "{\"error\":\"Modelo não carregado\"}";
      send(clientSocket, err.c_str(), err.length(), 0);
      return;
    }

    // Resize signal if needed
    if (g.size() != H->rows()) {
      if (g.size() < H->rows()) {
        Eigen::VectorXd padded = Eigen::VectorXd::Zero(H->rows());
        padded.head(g.size()) = g;
        g = padded;
      } else {
        g = g.head(H->rows());
      }
    }

    // Solve
    auto result = fast_cgnr_solve(g, *H, *H_T, 10);

    auto end_time = std::chrono::system_clock::now();
    std::time_t end_c = std::chrono::system_clock::to_time_t(end_time);
    std::stringstream end_ss;
    end_ss << std::put_time(std::localtime(&end_c), "%Y-%m-%d %H:%M:%S");

    std::time_t start_c = std::chrono::system_clock::to_time_t(start_time);
    std::stringstream start_ss;
    start_ss << std::put_time(std::localtime(&start_c), "%Y-%m-%d %H:%M:%S");

    // Build response
    std::string response = "{";
    response += "\"algorithm\":\"CGNR_CPP\",";
    response += "\"start_time\":\"" + start_ss.str() + "\",";
    response += "\"end_time\":\"" + end_ss.str() + "\",";
    response += "\"iterations\":" + std::to_string(result.iterations) + ",";
    response += "\"reduction_factor_C\":" + std::to_string(C_factor) + ",";
    response +=
        "\"execution_time_ms\":" + std::to_string(result.execution_time_ms) +
        ",";
    response += "\"image_pixels\":[";

    char buf[32];
    for (int i = 0; i < result.image.size(); ++i) {
      if (i > 0)
        response += ",";
      snprintf(buf, sizeof(buf), "%.6g", result.image[i]);
      response += buf;
    }
    response += "]}";

    send(clientSocket, response.c_str(), response.length(), 0);

    std::cout << "[OK] Solver=" << (int)result.execution_time_ms
              << "ms, Parse=" << (int)parse_time
              << "ms, Iters=" << result.iterations << "\n";

  } catch (const std::exception &e) {
    std::cerr << "[ERRO] " << e.what() << std::endl;
    std::string err = "{\"error\":\"" + std::string(e.what()) + "\"}";
    send(clientSocket, err.c_str(), err.length(), 0);
  }
}

int main() {
  std::cout << "=================================================="
            << std::endl;
  std::cout << "  SERVIDOR C++ - SOCKET RAW (Optimized)" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  // Ativar paralelismo Eigen
  // Para matrizes pequenas (60x60), threads causam overhead. Manter 1 thread.
  Eigen::setNbThreads(1);

  // Carregar matrizes
  try {
    std::cout << "\n[INFO] Carregando matrizes..." << std::endl;

    load_matrix_smart("data/H-2.csv", H_30);
    H_30_T = H_30.transpose().eval();
    C_factor = calculate_reduction_factor(H_30);
    std::cout << "[OK] H_30: " << H_30.rows() << "x" << H_30.cols()
              << std::endl;

    load_matrix_smart("data/H-1.csv", H_60);
    H_60_T = H_60.transpose().eval();
    std::cout << "[OK] H_60: " << H_60.rows() << "x" << H_60.cols()
              << std::endl;

    if (std::filesystem::exists("data/H-large.csv.dense.bin") ||
        std::filesystem::exists("data/H-large.csv")) {
      std::cout << "[INFO] Carregando matriz LARGE (MKL)..." << std::endl;
      load_matrix_smart("data/H-large.csv", H_large_mkl);
      H_large_T_mkl = H_large_mkl.transpose().eval();
      std::cout << "[OK] H_large_mkl: " << H_large_mkl.rows() << "x"
                << H_large_mkl.cols() << std::endl;
    }

    // Inicializar Winsock
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
      std::cerr << "[ERRO] Falha ao inicializar Winsock" << std::endl;
      return 1;
    }
#endif

    // Criar socket
    SocketType serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET_VAL) {
      std::cerr << "[ERRO] Falha ao criar socket" << std::endl;
      return 1;
    }

    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
    setsockopt(serverSocket, IPPROTO_TCP, TCP_NODELAY, (const char *)&opt,
               sizeof(opt));

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(8080);

    if (bind(serverSocket, (sockaddr *)&serverAddr, sizeof(serverAddr)) ==
        SOCKET_ERROR_VAL) {
      std::cerr << "[ERRO] Falha ao fazer bind na porta 8080" << std::endl;
      CLOSE_SOCKET(serverSocket);
      return 1;
    }

    if (listen(serverSocket, 10) == SOCKET_ERROR_VAL) {
      std::cerr << "[ERRO] Falha ao fazer listen" << std::endl;
      CLOSE_SOCKET(serverSocket);
      return 1;
    }

    std::cout << "\n[READY] Servidor pronto na porta 8080" << std::endl;
    std::cout << "[INFO] Aguardando conexões..." << std::endl;

    while (true) {
      sockaddr_in clientAddr{};
      int clientLen = sizeof(clientAddr);
      SocketType clientSocket = accept(serverSocket, (sockaddr *)&clientAddr,
                                       (socklen_t *)&clientLen);

      if (clientSocket != INVALID_SOCKET_VAL) {
        handle_client(clientSocket);
        CLOSE_SOCKET(clientSocket);
      }
    }

    CLOSE_SOCKET(serverSocket);
#ifdef _WIN32
    WSACleanup();
#endif

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[FATAL] " << e.what() << std::endl;
    return 1;
  }
}
