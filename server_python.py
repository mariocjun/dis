"""
Servidor Python para Reconstrução de Imagem
Usa sockets raw para comparação justa com C++.
"""
import socket
import json
import numpy as np
import time
import sys
import os

# --- Configuração ---
HOST = '127.0.0.1'
PORT = 8081
BUFFER_SIZE = 4096 * 4096

# --- Variáveis Globais ---
H_30 = None
H_60 = None
H_large = None
H_parse = None
H_30_T = None
H_60_T = None
H_large_T = None
H_parse_T = None
C_factor = 0.0


def load_matrix_smart(csv_filename):
    """Carrega matriz. Se existir .npy, usa ele. Senão, converte CSV."""
    npy_filename = csv_filename + ".npy"
    
    if os.path.exists(npy_filename):
        print(f"[INFO] Carregando binário: {npy_filename}")
        return np.load(npy_filename)
    
    if not os.path.exists(csv_filename):
        print(f"[ERRO] Arquivo não encontrado: {csv_filename}")
        sys.exit(1)
    
    print(f"[INFO] Convertendo CSV para binário: {csv_filename}")
    mat = np.loadtxt(csv_filename, delimiter=',')
    np.save(npy_filename, mat)
    print(f"[INFO] Binário salvo: {npy_filename}")
    
    return mat


def calculate_spectral_norm(H, iterations=10):
    """Calcula norma espectral de H^T * H."""
    M, N = H.shape
    b_k = np.random.rand(N)
    b_k = b_k / np.linalg.norm(b_k)
    
    for _ in range(iterations):
        y = H @ b_k
        z = H.T @ y
        b_k = z / np.linalg.norm(z)
    
    y = H @ b_k
    return np.dot(y, y)


def cgnr_solver(g, H, H_T, iterations=10):
    """
    CGNR Solver com Regularização de Tikhonov.
    Termina quando erro < 1e-4 ou atingir max iterações.
    """
    start_time = time.time()
    
    M, N = H.shape
    # print(f"DEBUG: Solving {M}x{N}. Iterations={iterations}")
    
    f = np.zeros(N)
    r = g - H @ f
    z = H_T @ r
    p = z.copy()
    
    lambda_reg = np.max(np.abs(z)) * 0.10
    if lambda_reg < 1e-9:
        lambda_reg = 1e-9
    
    previous_residual_norm = np.linalg.norm(r)
    final_iter = iterations
    
    for k in range(iterations):
        w = H @ p
        norm_z_sq = np.dot(z, z)
        norm_w_sq = np.dot(w, w)
        norm_p_sq = np.dot(p, p)
        
        denominator = norm_w_sq + lambda_reg * norm_p_sq
        
        if denominator < 1e-15:
            final_iter = k + 1
            break
        
        alpha = norm_z_sq / denominator
        f = f + alpha * p
        r = r - alpha * w
        
        current_residual_norm = np.linalg.norm(r)
        
        if k > 0:
            epsilon = abs(current_residual_norm - previous_residual_norm)
            if epsilon < 1e-4:
                final_iter = k + 1
                break
        
        previous_residual_norm = current_residual_norm
        z_next = (H_T @ r) - (lambda_reg * f)
        norm_z_next_sq = np.dot(z_next, z_next)
        beta = norm_z_next_sq / norm_z_sq
        p = z_next + beta * p
        z = z_next
    
    return f, time.time() - start_time, final_iter


def handle_client(conn, addr):
    """Processa uma conexão de cliente."""
    try:
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Receber dados
        data = b""
        while True:
            packet = conn.recv(BUFFER_SIZE)
            if not packet:
                break
            data += packet
            if packet.strip().endswith(b'}'):
                break
        
        if not data:
            return
        
        # Parse request
        request = json.loads(data.decode('utf-8'))
        model_size = request.get('model_size')
        signal_g = np.array(request.get('signal_g'))
        
        # Selecionar matriz
        if model_size == "30x30":
            H, H_T = H_30, H_30_T
        elif model_size == "large":
            H, H_T = H_large, H_large_T
            print(f"DEBUG: Large Model Selected. H shape: {H.shape}")
        elif model_size == "parse":
            H, H_T = H_parse, H_parse_T
        else:
            H, H_T = H_60, H_60_T
        
        if H is None:
            conn.sendall(json.dumps({"error": "Modelo não carregado"}).encode())
            return
        
        # Executar solver
        image_f, exec_time, iters = cgnr_solver(signal_g, H, H_T)
        
        # Resposta
        response = {
            "algorithm": "CGNR_PYTHON",
            "start_time": start_time_str,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": iters,
            "image_pixels": image_f.tolist(),
            "reduction_factor_C": C_factor,
            "execution_time_ms": exec_time * 1000
        }
        
        conn.sendall(json.dumps(response).encode())
        print(f"[OK] Solver={int(exec_time*1000)}ms, Iters={iters}")
        
    except Exception as e:
        print(f"[ERRO] {e}")
    finally:
        conn.close()


def main():
    global H_30, H_60, H_30_T, H_60_T, C_factor
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    print("=" * 50)
    print("  SERVIDOR PYTHON - SOCKET RAW")
    print("=" * 50)
    
    # Carregar matrizes
    print("\n[INFO] Carregando matrizes...")
    
    h_30_path = os.path.join(data_dir, "H-2.csv")
    H_30 = load_matrix_smart(h_30_path)
    H_30_T = H_30.T.copy()
    C_factor = calculate_spectral_norm(H_30)
    print(f"[OK] H_30: {H_30.shape}")
    
    h_60_path = os.path.join(data_dir, "H-1.csv")
    H_60 = load_matrix_smart(h_60_path)
    H_60_T = H_60.T.copy()
    print(f"[OK] H_60: {H_60.shape}")

    # Load Large if exists
    h_large_path = os.path.join(data_dir, "H-large.csv")
    if os.path.exists(h_large_path + ".npy") or os.path.exists(h_large_path):
        H_large = load_matrix_smart(h_large_path)
        H_large_T = H_large.T.copy()
        print(f"[OK] H_large: {H_large.shape}")
        
    # Load Parse
    h_parse_path = os.path.join(data_dir, "H-parse.csv")
    if os.path.exists(h_parse_path + ".npy") or os.path.exists(h_parse_path):
        H_parse = load_matrix_smart(h_parse_path)
        H_parse_T = H_parse.T.copy()
        print(f"[OK] H_parse: {H_parse.shape}")

    # Iniciar servidor
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind((HOST, PORT))
        s.listen(10)
        
        print(f"\n[READY] Servidor pronto na porta {PORT}")
        print("[INFO] Aguardando conexões...")
        
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)


if __name__ == "__main__":
    main()