import subprocess
import time
import sys
import os
import platform
import socket

def is_port_open(host, port):
    """Verifica se a porta está aberta (servidor pronto)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def wait_for_server(port, name, timeout=180): # Timeout alto para carregar H-1.csv
    print(f"[ORCHESTRATOR] Aguardando {name} na porta {port}...")
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open('127.0.0.1', port):
            print(f"[ORCHESTRATOR] {name} está pronto!")
            return True
        time.sleep(1)
    print(f"[ORCHESTRATOR] Timeout esperando por {name}.")
    return False

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(root_dir, "build")
    
    # 1. Compilação (C++)
    print("\n=== PASSO 1: COMPILAÇÃO (C++) ===")
    if not os.path.exists(build_dir): os.makedirs(build_dir)
    
    # Configura e Compila
    if platform.system() == "Windows":
        subprocess.call(f"cmake -S . -B build", shell=True, cwd=root_dir)
        subprocess.call(f"cmake --build build --config Release", shell=True, cwd=root_dir)
        exe_path = os.path.join(build_dir, "Release", "UltrasoundBenchmark.exe")
    else:
        subprocess.call(f"cmake -S . -B build -DCMAKE_BUILD_TYPE=Release", shell=True, cwd=root_dir)
        subprocess.call(f"cmake --build build", shell=True, cwd=root_dir)
        exe_path = os.path.join(build_dir, "UltrasoundBenchmark")

    if not os.path.exists(exe_path):
        print(f"[ERRO] Executável não encontrado em: {exe_path}")
        sys.exit(1)

    # 2. Iniciar Servidores
    print("\n=== PASSO 2: INICIANDO SERVIDORES ===")
    print("Nota: O carregamento das matrizes (especialmente H-1.csv de 680MB) pode levar tempo na primeira vez.")
    print("      Nas próximas vezes será instantâneo (usando .bin/.npy).")
    
    # Inicia C++ (sem bloquear o terminal)
    cpp_server = subprocess.Popen([exe_path], cwd=root_dir)
    
    # Inicia Python (sem bloquear o terminal)
    py_server_path = os.path.join(root_dir, "server_python.py")
    py_server = subprocess.Popen([sys.executable, py_server_path], cwd=root_dir)

    try:
        # Aguarda os servidores subirem (TCP Port Check)
        if not wait_for_server(8080, "Servidor C++"): raise Exception("Falha ao iniciar C++")
        if not wait_for_server(8081, "Servidor Python"): raise Exception("Falha ao iniciar Python")

        # 3. Rodar Cliente
        print("\n=== PASSO 3: EXECUTANDO BATERIA DE TESTES ===")
        client_path = os.path.join(root_dir, "client.py")
        
        # Roda o cliente e espera ele terminar
        subprocess.run([sys.executable, client_path], cwd=root_dir)

    except KeyboardInterrupt:
        print("\n[ORCHESTRATOR] Interrompido pelo usuário.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        # 4. Limpeza (Matar processos)
        print("\n=== PASSO 4: LIMPEZA ===")
        print("[ORCHESTRATOR] Encerrando servidores...")
        cpp_server.terminate()
        py_server.terminate()
        # Garante que morreram
        time.sleep(1)
        if cpp_server.poll() is None: cpp_server.kill()
        if py_server.poll() is None: py_server.kill()
        print("[ORCHESTRATOR] Finalizado.")

if __name__ == "__main__":
    main()