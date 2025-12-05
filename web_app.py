import os
import time
import sys
import platform
import socket
import json
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import subprocess
import threading

app = Flask(__name__)
# Use threading mode for simplicity and Windows compatibility
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*", logger=True, engineio_logger=True)

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
STATIC_RESULTS_DIR = os.path.join(ROOT_DIR, 'static', 'results')
SERVER_CPP_PORT = 8080
SERVER_PY_PORT = 8081

# Ensure results directory exists
if not os.path.exists(STATIC_RESULTS_DIR):
    os.makedirs(STATIC_RESULTS_DIR)

# --- Helper Functions ---

def log(message):
    print(f"[WEB-APP] {message}", flush=True)
    socketio.emit('log', {'data': message})

def kill_ports(ports):
    """
    Kills any process listening on the specified ports.
    """
    for port in ports:
        try:
            if platform.system() == "Windows":
                # Safer one-liner to find and kill
                cmd = f"FOR /F \"tokens=5\" %P IN ('netstat -a -n -o ^| findstr :{port}') DO taskkill /F /PID %P"
                subprocess.call(cmd, shell=True)
            else:
                subprocess.call(f"fuser -k {port}/tcp", shell=True)
        except Exception as e:
            log(f"Aviso ao limpar porta {port}: {e}")

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def wait_for_server(port, name, timeout=180):
    log(f"Aguardando {name} na porta {port}...")
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open('127.0.0.1', port):
            log(f"{name} está pronto!")
            return True
        time.sleep(1)
    log(f"Timeout esperando por {name}.")
    return False

def load_signal(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        log(f"Arquivo não encontrado: {filename}")
        return None
    return np.loadtxt(filepath, delimiter=',')

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

def send_request(port, model_size, signal_g):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(120.0)
            s.connect(('127.0.0.1', port))
            
            payload = {
                "model_size": model_size,
                "signal_g": signal_g.tolist()
            }
            s.sendall(json.dumps(payload).encode('utf-8'))
            
            data = b""
            while True:
                packet = s.recv(4096*4096)
                if not packet: break
                data += packet
                if packet.strip().endswith(b'}'): break
            
            return json.loads(data.decode('utf-8'))
    except Exception as e:
        log(f"Erro de comunicação com porta {port}: {e}")
        return None

def generate_heatmap(pixels, model_size, filename):
    """
    Gera e salva o heatmap da imagem reconstruída.
    Melhorias: Grayscale, Thresholding, Normalização.
    """
    dim = 30 if model_size == "30x30" else 60
    
    # Reshape
    try:
        img_matrix = np.array(pixels).reshape((dim, dim))
    except ValueError:
        log(f"Erro ao fazer reshape da imagem {filename}. Tamanho recebido: {len(pixels)}")
        return None

    # 1. Thresholding: Remove negative noise
    img_matrix[img_matrix < 0] = 0
    
    # 2. Normalization [0, 1]
    max_val = img_matrix.max()
    if max_val > 0:
        img_matrix = img_matrix / max_val

    # Plot using OO API (Thread Safe)
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    # 3. Grayscale Colormap
    ax.imshow(img_matrix, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax.axis('off')
    
    # Save
    save_path = os.path.join(STATIC_RESULTS_DIR, filename)
    # Remove file if exists to ensure freshness
    if os.path.exists(save_path):
        os.remove(save_path)
        
    canvas = FigureCanvas(fig)
    canvas.print_png(save_path)
    
    return f"static/results/{filename}"

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

# --- Socket Events ---

@socketio.on('connect')
def handle_connect():
    log(f"Cliente conectado: {socket.gethostname()}")

@socketio.on('disconnect')
def handle_disconnect():
    log("Cliente desconectado.")

@socketio.on('start_benchmark')
def handle_start_benchmark():
    log("Recebido comando de início. Preparando benchmark...")
    # Use socketio background task
    socketio.start_background_task(run_benchmark_task)

def run_benchmark_task():
    """Executes the benchmark in a separate thread to avoid blocking the server."""
    cpp_server = None
    py_server = None

    try:
        log("Iniciando tarefa de Benchmark...")
        
        # 0. Limpar Portas (Safety Check)
        ports_to_check = [SERVER_CPP_PORT, SERVER_PY_PORT]
        active_ports = [p for p in ports_to_check if is_port_open('127.0.0.1', p)]
        
        if active_ports:
            log(f"Detectadas portas ocupadas: {active_ports}. Tentando liberar...")
            kill_ports(active_ports)
            time.sleep(2)
        else:
            log("Portas 8080/8081 parecem livres.")
        
        # 1. Compilação
        log("Compilando projeto C++...")
        build_dir = os.path.join(ROOT_DIR, "build")
        if not os.path.exists(build_dir): os.makedirs(build_dir)
        
        cmd_config = "cmake -S . -B build"
        cmd_build = "cmake --build build --config Release"
        
        if platform.system() != "Windows":
            cmd_config += " -DCMAKE_BUILD_TYPE=Release"

        if subprocess.call(cmd_config, shell=True, cwd=ROOT_DIR) != 0:
            log("Erro na configuração do CMake.")
            return
            
        if subprocess.call(cmd_build, shell=True, cwd=ROOT_DIR) != 0:
            log("Erro na compilação.")
            return
        
        # Tenta localizar o executável em locais padrões (Ninja vs MSVC)
        possible_paths = []
        if platform.system() == "Windows":
            possible_paths.append(os.path.join(build_dir, "UltrasoundBenchmark.exe"))            # Ninja / Single Config
            possible_paths.append(os.path.join(build_dir, "Release", "UltrasoundBenchmark.exe")) # MSVC / Multi Config
            possible_paths.append(os.path.join(build_dir, "Debug", "UltrasoundBenchmark.exe"))   # Debug fallback
        else:
            possible_paths.append(os.path.join(build_dir, "UltrasoundBenchmark"))

        exe_path = None
        for p in possible_paths:
            if os.path.exists(p):
                exe_path = p
                break
        
        if not exe_path:
            log(f"Erro: Executável C++ não encontrado. Tentados: {possible_paths}")
            return

        # 2. Iniciar Servidores
        log("Iniciando servidores...")
        
        # Start Py Server
        py_server = subprocess.Popen([sys.executable, os.path.join(ROOT_DIR, "server_python.py")], cwd=ROOT_DIR)
        
        # Start C++ Server
        cpp_server = subprocess.Popen([exe_path], cwd=ROOT_DIR)
        
        
        if not wait_for_server(SERVER_CPP_PORT, "C++") or not wait_for_server(SERVER_PY_PORT, "Python"):
            raise Exception("Falha ao iniciar servidores")

        # 3. Executar Cenários
        scenarios = [
            {"model": "30x30", "file": "g-30x30-1.csv"},
            {"model": "30x30", "file": "g-30x30-2.csv"},
            {"model": "30x30", "file": "A-30x30-1.csv"},
            {"model": "60x60", "file": "G-1.csv"},
            {"model": "60x60", "file": "G-2.csv"},
            {"model": "60x60", "file": "A-60x60-1.csv"}
        ]
        
        for task in scenarios:
            filename = task["file"]
            model = task["model"]
            
            log(f"Processando {filename} ({model})...")
            
            # Load & Gain
            raw_signal = load_signal(filename)
            if raw_signal is None: continue
            
            processed_signal = apply_gain(raw_signal)
            
            # Request C++
            start = time.time()
            resp_cpp = send_request(SERVER_CPP_PORT, model, processed_signal)
            time_cpp = time.time() - start if resp_cpp else 0
            
            # Request Python
            start = time.time()
            resp_py = send_request(SERVER_PY_PORT, model, processed_signal)
            time_py = time.time() - start if resp_py else 0
            
            # Generate Images
            img_cpp_path = ""
            img_py_path = ""
            
            if resp_cpp:
                img_cpp_path = generate_heatmap(resp_cpp['image_pixels'], model, f"cpp_{filename}.png")
            
            if resp_py:
                img_py_path = generate_heatmap(resp_py['image_pixels'], model, f"py_{filename}.png")
            
            # Send Result
            result_data = {
                "filename": filename,
                "model": model,
                "time_cpp": time_cpp,
                "time_py": time_py,
                "speedup": time_py / time_cpp if time_cpp > 0 else 0,
                "img_cpp": img_cpp_path,
                "img_py": img_py_path,
                "iters_cpp": resp_cpp.get('iterations', 0) if resp_cpp else 0,
                "iters_py": resp_py.get('iterations', 0) if resp_py else 0
            }
            
            socketio.emit('new_result', result_data)
            time.sleep(0.1) # UI refresh

    except Exception as e:
        log(f"Erro durante benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log("Encerrando servidores...")
        if cpp_server:
            try:
                cpp_server.terminate()
            except: pass
        if py_server:
            try:
                py_server.terminate()
            except: pass
            
        # Hard kill if needed
        time.sleep(1)
        if cpp_server and cpp_server.poll() is None:
            try: cpp_server.kill()
            except: pass
        if py_server and py_server.poll() is None:
            try: py_server.kill()
            except: pass
        
        log("Benchmark Finalizado.")
        socketio.emit('benchmark_finished')

if __name__ == '__main__':
    print("Iniciando servidor Web (Threading Mode)...", flush=True)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
