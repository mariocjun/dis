#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
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

// --- Global Variables ---
Eigen::SparseMatrix<double> H_30;
Eigen::SparseMatrix<double> H_60;
double C_factor = 0.0;

// --- Helper Functions ---

// Carrega matriz de forma inteligente: Tenta BIN primeiro, senão CSV e salva
// BIN.
void load_matrix_smart(const std::string &csv_path, int cols,
                       Eigen::SparseMatrix<double> &H) {
  std::string bin_path = csv_path + ".sparse.bin";

  if (std::filesystem::exists(bin_path)) {
    std::cout << "[INFO] Carregando binario rapido: " << bin_path << "..."
              << std::endl;
    H = loadSparseMatrix(bin_path);
  } else {
    if (!std::filesystem::exists(csv_path)) {
      throw std::runtime_error("Arquivo CSV nao encontrado: " + csv_path);
    }
    std::cout << "[INFO] Binario nao encontrado. Convertendo CSV (lento): "
              << csv_path << "..." << std::endl;
    H = convertCsvToSparse(csv_path, cols);

    std::cout << "[INFO] Salvando binario para uso futuro..." << std::endl;
    saveSparseMatrix(H, bin_path);
    std::cout << "[INFO] Binario salvo em: " << bin_path << std::endl;
  }
}

double
calculate_spectral_norm_power_iteration(const Eigen::SparseMatrix<double> &H,
                                        int iterations = 20) {
  // Calculates ||H^T H||_2 = lambda_max(H^T * H)
  Eigen::Index N = H.cols();
  Eigen::VectorXd b_k = Eigen::VectorXd::Random(N);
  b_k.normalize();

  for (int i = 0; i < iterations; ++i) {
    Eigen::VectorXd y = H * b_k;
    Eigen::VectorXd z = H.transpose() * y;
    b_k = z.normalized();
  }

  Eigen::VectorXd y = H * b_k;
  double lambda_max = y.squaredNorm();
  return lambda_max;
}

double calculate_reduction_factor(const Eigen::SparseMatrix<double> &H) {
  // C = ||H^T H||_2
  return calculate_spectral_norm_power_iteration(H);
}

void send_response(SocketType clientSocket, const json &response) {
  std::string response_str = response.dump();
  send(clientSocket, response_str.c_str(), (int)response_str.length(), 0);
}

void handle_client(SocketType clientSocket) {
  const int buffer_size = 4096 * 16;
  std::vector<char> buffer(buffer_size);
  std::string received_data;

  while (true) {
    int bytesReceived = recv(clientSocket, buffer.data(), buffer_size, 0);
    if (bytesReceived > 0) {
      received_data.append(buffer.data(), bytesReceived);
      try {
        json request = json::parse(received_data);
        break;
      } catch (const json::parse_error &e) {
        continue;
      }
    } else if (bytesReceived == 0) {
      return;
    } else {
      return;
    }
  }

  try {
    json request = json::parse(received_data);
    std::string model_size = request["model_size"];
    std::vector<double> signal_vec = request["signal_g"];

    Eigen::VectorXd g_signal =
        Eigen::Map<Eigen::VectorXd>(signal_vec.data(), signal_vec.size());

    std::cout << "[INFO] Request: Model=" << model_size
              << ", Signal Size=" << g_signal.size() << std::endl;

    Eigen::SparseMatrix<double> *H = nullptr;
    if (model_size == "30x30")
      H = &H_30;
    else if (model_size == "60x60")
      H = &H_60;

    if (H == nullptr || H->rows() == 0) {
      json error_msg = {{"error", "Invalid model size or model not loaded"}};
      send_response(clientSocket, error_msg);
      return;
    }

    int img_dim = (model_size == "30x30") ? 30 : 60;

    auto result = run_cgnr_solver_epsilon_save_iters(g_signal, *H, 1e-4, 10, "",
                                                     "", img_dim, img_dim);

    std::vector<double> image_pixels(result.image.data(),
                                     result.image.data() + result.image.size());

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");

    json response = {{"algorithm", "CGNR_CPP"},
                     {"start_time", "N/A"},
                     {"end_time", ss.str()},
                     {"iterations", result.iterations},
                     {"image_pixels", image_pixels},
                     {"reduction_factor_C", C_factor},
                     {"execution_time_ms", result.execution_time_ms}};

    send_response(clientSocket, response);
    std::cout << "[INFO] Response sent." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[ERRO] Error processing request: " << e.what() << std::endl;
    json error_msg = {{"error", e.what()}};
    send_response(clientSocket, error_msg);
  }
}

int main() {
  std::cout << "[INFO] Iniciando Servidor C++..." << std::endl;

  // 1. Load Matrices (Smart Load)
  try {
    std::string h30_path = "data/H-2.csv";
    load_matrix_smart(h30_path, 900, H_30);
    C_factor = calculate_reduction_factor(H_30);
    std::cout << "[INFO] H_30 carregada. C_factor: " << C_factor << std::endl;

    std::string h60_path = "data/H-1.csv";
    load_matrix_smart(h60_path, 3600, H_60);
    if (C_factor == 0.0)
      C_factor = calculate_reduction_factor(H_60);
    std::cout << "[INFO] H_60 carregada." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[ERRO FATAL] Falha ao carregar matrizes: " << e.what()
              << std::endl;
    return 1;
  }

  // 2. Initialize Winsock (Windows Only)
  int iResult;
#ifdef _WIN32
  WSADATA wsaData;
  iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if (iResult != 0)
    return 1;
#else
  iResult = 0;
#endif

  struct addrinfo *result = NULL, hints;
#ifdef _WIN32
  ZeroMemory(&hints, sizeof(hints));
#else
  std::memset(&hints, 0, sizeof(hints));
#endif
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_PASSIVE;

  getaddrinfo(NULL, "8080", &hints, &result);
  SocketType ListenSocket =
      socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
  freeaddrinfo(result);
  listen(ListenSocket, SOMAXCONN);

  std::cout << "[READY] C++ Server ouvindo na porta 8080..." << std::endl;

  SocketType ClientSocket = INVALID_SOCKET_VAL;
  while (true) {
    ClientSocket = accept(ListenSocket, NULL, NULL);
    if (ClientSocket != INVALID_SOCKET_VAL) {
      handle_client(ClientSocket);
      CLOSE_SOCKET(ClientSocket);
    }
  }

  CLOSE_SOCKET(ListenSocket);
#ifdef _WIN32
  WSACleanup();
#endif
  return 0;
}