import socket
import json
import numpy as np
import time
import os
import sys
import math

# --- Configuração ---
SERVER_CPP_HOST = '127.0.0.1'
SERVER_CPP_PORT = 8080
SERVER_PY_HOST = '127.0.0.1'
SERVER_PY_PORT = 8081
BUFFER_SIZE = 4096 * 4096

def load_signal(filename):
    """Carrega o sinal do arquivo CSV."""
    if not os.path.exists(filename):
        # Tenta procurar na pasta data se o caminho não for absoluto
        if os.path.exists(os.path.join('data', filename)):
            return np.loadtxt(os.path.join('data', filename), delimiter=',')
        print(f"[ERRO] Arquivo não encontrado: {filename}")
        return None
    return np.loadtxt(filename, delimiter=',')

def apply_gain(signal):
    """
    Aplica o ganho conforme enunciado: gamma_l = 100 + (1/20) * l * sqrt(l)
    """
    gained_signal = np.zeros_like(signal)
    for k in range(len(signal)):
        l = k + 1  # Índice l começa em 1
        gain = 100 + (1/20.0) * l * math.sqrt(l)
        gained_signal[k] = signal[k] * gain
    return gained_signal

def send_request(host, port, request_data):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(120.0) # Timeout alto para garantir processamento de matrizes grandes
            s.connect((host, port))
            s.sendall(json.dumps(request_data).encode('utf-8'))
            
            data = b""
            while True:
                packet = s.recv(BUFFER_SIZE)
                if not packet: break
                data += packet
                if packet.strip().endswith(b'}'): break
            
            return json.loads(data.decode('utf-8'))
    except Exception as e:
        print(f"[FALHA DE CONEXÃO] {host}:{port} - {e}")
        return None

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # --- LISTA DE TAREFAS (Os 6 arquivos da sua pasta) ---
    scenarios = [
        # Modelo 30x30 (Usa H-2.csv)
        {"model": "30x30", "file": "g-30x30-1.csv"},
        {"model": "30x30", "file": "g-30x30-2.csv"},
        {"model": "30x30", "file": "A-30x30-1.csv"},
        
        # Modelo 60x60 (Usa H-1.csv)
        {"model": "60x60", "file": "G-1.csv"},
        {"model": "60x60", "file": "G-2.csv"},
        {"model": "60x60", "file": "A-60x60-1.csv"}
    ]

    print("\n" + "=" * 100)
    print(f"{'ARQUIVO':<20} | {'MODELO':<8} | {'C++ (s)':<10} | {'PY (s)':<10} | {'SPEEDUP':<10} | {'ITERS (C++/Py)':<15}")
    print("=" * 100)

    results = []

    for task in scenarios:
        filename = task["file"]
        model = task["model"]
        filepath = os.path.join(data_dir, filename)

        # 1. Carregar Sinal
        raw_signal = load_signal(filepath)
        
        if raw_signal is None:
            print(f"{filename:<20} | {model:<8} | ARQUIVO NÃO ENCONTRADO - Pulando")
            continue

        # 2. Aplicar Ganho (Regra de Negócio)
        processed_signal = apply_gain(raw_signal)

        payload = {
            "model_size": model,
            "signal_g": processed_signal.tolist()
        }

        # 3. Executar no C++
        start = time.time()
        resp_cpp = send_request(SERVER_CPP_HOST, SERVER_CPP_PORT, payload)
        time_cpp = time.time() - start if resp_cpp else None

        # 4. Executar no Python
        start = time.time()
        resp_py = send_request(SERVER_PY_HOST, SERVER_PY_PORT, payload)
        time_py = time.time() - start if resp_py else None

        # 5. Formatar Saída
        cpp_str = f"{time_cpp:.4f}" if time_cpp else "FAIL"
        py_str = f"{time_py:.4f}" if time_py else "FAIL"
        
        speedup_str = "-"
        iters_str = "-"
        
        if time_cpp and time_py and time_cpp > 0:
            speedup = time_py / time_cpp
            speedup_str = f"{speedup:.2f}x"
            
            it_cpp = resp_cpp.get("iterations", "?")
            it_py = resp_py.get("iterations", "?")
            iters_str = f"{it_cpp} / {it_py}"

        print(f"{filename:<20} | {model:<8} | {cpp_str:<10} | {py_str:<10} | {speedup_str:<10} | {iters_str:<15}")
        
        results.append({
            "file": filename,
            "cpp": time_cpp,
            "py": time_py
        })
        
        # Pequena pausa para garantir estabilidade dos sockets
        time.sleep(0.5)

    # --- RELATÓRIO FINAL ---
    print("=" * 100)
    print("RESUMO FINAL DA BATERIA DE TESTES")
    valid_runs = [r for r in results if r['cpp'] and r['py']]
    
    if valid_runs:
        avg_cpp = sum(r['cpp'] for r in valid_runs) / len(valid_runs)
        avg_py = sum(r['py'] for r in valid_runs) / len(valid_runs)
        
        print(f"Total de Imagens Processadas: {len(valid_runs)}")
        print(f"Tempo Médio C++:    {avg_cpp:.4f} s")
        print(f"Tempo Médio Python: {avg_py:.4f} s")
        
        if avg_cpp > 0:
            print(f"Speedup Médio Global: {avg_py/avg_cpp:.2f}x (C++ é mais rápido)")
    else:
        print("Nenhum teste foi concluído com sucesso em ambos os servidores.")
    print("=" * 100)

if __name__ == "__main__":
    main()