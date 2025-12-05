import socket
import json
import numpy as np
import time
import sys
import os
import math

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 8081
BUFFER_SIZE = 4096 * 4096 

# --- Global Variables ---
H_30 = None
H_60 = None
C_factor = 0.0

def load_matrix_smart(csv_filename):
    """
    Carrega a matriz. Se existir .npy (binário), carrega ele.
    Se não, carrega CSV, salva .npy e retorna.
    """
    npy_filename = csv_filename + ".npy"
    
    if os.path.exists(npy_filename):
        print(f"[INFO] Carregando binario rapido: {npy_filename}...")
        return np.load(npy_filename)
    
    if not os.path.exists(csv_filename):
        print(f"[ERROR] Arquivo nao encontrado: {csv_filename}")
        sys.exit(1)
        
    print(f"[INFO] Binario nao encontrado. Convertendo CSV (lento): {csv_filename}...")
    # Load CSV
    mat = np.loadtxt(csv_filename, delimiter=',')
    
    print(f"[INFO] Salvando binario .npy para uso futuro...")
    np.save(npy_filename, mat)
    print(f"[INFO] Binario salvo.")
    
    return mat

def calculate_spectral_norm_power_iteration(H, iterations=10):
    """
    Calcula a norma espectral de H^T * H.
    """
    M, N = H.shape
    b_k = np.random.rand(N)
    b_k = b_k / np.linalg.norm(b_k)
    
    for _ in range(iterations):
        y = H @ b_k
        z = H.T @ y
        b_k = z / np.linalg.norm(z)
        
    y = H @ b_k
    lambda_max = np.dot(y, y)
    return lambda_max

def calculate_reduction_factor(H):
    return calculate_spectral_norm_power_iteration(H)

def cgnr_solver(g, H, iterations=10):
    """
    CGNR Solver com Regularização de Tikhonov.
    """
    start_time = time.time()
    M, N = H.shape
    f = np.zeros(N)
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    
    lambda_reg = np.max(np.abs(z)) * 0.0001
    if lambda_reg < 1e-9: lambda_reg = 1e-9
        
    previous_residual_norm = np.linalg.norm(r)
    
    final_iter = iterations # Assume maximo se nao parar antes

    for k in range(iterations):
        w = H @ p
        norm_z_sq = np.dot(z, z)
        norm_w_sq = np.dot(w, w)
        norm_p_sq = np.dot(p, p)
        
        denominator = norm_w_sq + lambda_reg * norm_p_sq
        
        if denominator < 1e-15:
            print(f"[WARN] Denominator close to zero at iter {k+1}")
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
        z_next = (H.T @ r) - (lambda_reg * f)
        norm_z_next_sq = np.dot(z_next, z_next)
        beta = norm_z_next_sq / norm_z_sq
        p = z_next + beta * p
        z = z_next
        
    return f, time.time() - start_time, final_iter

def handle_client(conn, addr):
    try:
        data = b""
        while True:
            packet = conn.recv(BUFFER_SIZE)
            if not packet: break
            data += packet
            if packet.strip().endswith(b'}'): break
                
        if not data: return

        request = json.loads(data.decode('utf-8'))
        model_size = request.get('model_size')
        signal_g = np.array(request.get('signal_g'))
        
        H = H_30 if model_size == "30x30" else H_60
        if H is None:
            conn.sendall(json.dumps({"error": "Model not loaded"}).encode('utf-8'))
            return

        image_f, exec_time, iters = cgnr_solver(signal_g, H)
        
        response = {
            "algorithm": "CGNR_PYTHON",
            "start_time": "N/A",
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "iterations": iters,
            "image_pixels": image_f.tolist(),
            "reduction_factor_C": C_factor,
            "execution_time_ms": exec_time * 1000
        }
        conn.sendall(json.dumps(response).encode('utf-8'))
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        conn.close()

def main():
    global H_30, H_60, C_factor
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    print("[INFO] Iniciando Servidor Python...")

    # Carrega H-2 (30x30)
    h_30_path = os.path.join(data_dir, "H-2.csv") 
    H_30 = load_matrix_smart(h_30_path)
    C_factor = calculate_reduction_factor(H_30)
    print(f"[INFO] H_30 carregada. C_factor: {C_factor:.4e}")

    # Carrega H-1 (60x60)
    h_60_path = os.path.join(data_dir, "H-1.csv")
    H_60 = load_matrix_smart(h_60_path)
    if C_factor == 0.0: C_factor = calculate_reduction_factor(H_60)
    print(f"[INFO] H_60 carregada.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[READY] Python Server ouvindo na porta {PORT}")
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)

if __name__ == "__main__":
    main()