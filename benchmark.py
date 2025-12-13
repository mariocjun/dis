#!/usr/bin/env python3
"""
🏆 BENCHMARK DE RECONSTRUÇÃO DE IMAGEM ULTRASSOM
Compara performance entre servidores C++ e Python.
"""

# IMPORTANTE: Configurar backend ANTES de qualquer import matplotlib
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI (evita erros de threading)

import os
import sys
import time
import json
import socket
import random
import argparse
import subprocess
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.live import Live
from rich import box

console = Console()

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CPP_PORT = 8080
PYTHON_PORT = 8081
HOST = '127.0.0.1'
BUFFER_SIZE = 4096 * 4096

SIGNALS = [
    {"file": "G-1.csv", "model": "60x60"},
    {"file": "G-2.csv", "model": "60x60"},
    {"file": "g-30x30-1.csv", "model": "30x30"},
    {"file": "g-30x30-2.csv", "model": "30x30"},
    {"file": "g-large.csv", "model": "large"},
    {"file": "g-parse.csv", "model": "parse"},
]

# ============================================================
# RESOURCE MONITOR (CPU, RAM, Threads)
# ============================================================
class ResourceMonitor:
    """Monitora CPU, memória e threads de um processo."""
    
    def __init__(self, pid: int, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = {
            "cpu_percent": [],
            "memory_mb": [],
            "num_threads": [],
            "timestamps": []
        }
        
    def _monitor_loop(self):
        try:
            proc = psutil.Process(self.pid)
            proc.cpu_percent()  # Primeira chamada para inicializar
            time.sleep(0.1)
            
            while self.running:
                try:
                    cpu = proc.cpu_percent()
                    mem = proc.memory_info().rss / (1024 * 1024)  # MB
                    threads = proc.num_threads()
                    
                    self.samples["cpu_percent"].append(cpu)
                    self.samples["memory_mb"].append(mem)
                    self.samples["num_threads"].append(threads)
                    self.samples["timestamps"].append(time.time())
                    
                    time.sleep(self.interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def get_stats(self) -> dict:
        if not self.samples["cpu_percent"]:
            return {"cpu_avg": 0, "cpu_max": 0, "mem_avg": 0, "mem_max": 0, "threads_avg": 0, "threads_max": 0}
        
        return {
            "cpu_avg": sum(self.samples["cpu_percent"]) / len(self.samples["cpu_percent"]),
            "cpu_max": max(self.samples["cpu_percent"]),
            "mem_avg": sum(self.samples["memory_mb"]) / len(self.samples["memory_mb"]),
            "mem_max": max(self.samples["memory_mb"]),
            "threads_avg": sum(self.samples["num_threads"]) / len(self.samples["num_threads"]),
            "threads_max": max(self.samples["num_threads"]),
        }

# ============================================================
# LOGGER DETALHADO
# ============================================================
class DetailedLogger:
    """Registra TODOS os detalhes da execução."""
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.entries = []
        self.start_time = datetime.now()
        
    def log(self, category: str, message: str, data: dict = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "category": category,
            "message": message,
            "data": data or {}
        }
        self.entries.append(entry)
        
        if self.filepath:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(f"[{entry['elapsed_ms']:.0f}ms] [{category}] {message}\n")
                if data:
                    for k, v in data.items():
                        f.write(f"    {k}: {v}\n")
    
    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

logger = None

# ============================================================
# COMUNICAÇÃO SOCKET
# ============================================================
def send_request(host: str, port: int, payload: dict, timeout: float = 60.0) -> Tuple[Optional[dict], float]:
    """Envia requisição via socket e retorna (resposta, latência_ms)."""
    start = time.perf_counter()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(timeout)
            s.connect((host, port))
            
            data_json = json.dumps(payload)
            s.sendall(data_json.encode('utf-8'))
            
            data = b""
            while True:
                packet = s.recv(BUFFER_SIZE)
                if not packet:
                    break
                data += packet
                if packet.strip().endswith(b'}'):
                    break
            
            latency_ms = (time.perf_counter() - start) * 1000
            response = json.loads(data.decode('utf-8'))
            return response, latency_ms
            
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return None, latency_ms

def apply_gain(signal: np.ndarray) -> np.ndarray:
    """Aplica ganho: gamma_l = 100 + (1/20) * l * sqrt(l)"""
    l = np.arange(1, len(signal) + 1)
    gain = 100 + (1/20.0) * l * np.sqrt(l)
    return signal * gain

def load_signal(filename: str, data_dir: str = "data") -> Optional[np.ndarray]:
    """Carrega sinal do arquivo CSV."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        return None
    return np.loadtxt(filepath, delimiter=',')


def save_reconstructed_image(
    image_pixels: list,
    model_size: str,
    server_type: str,
    solver_time_ms: float,
    iterations: int,
    output_dir: str = "results/images"
) -> str:
    """
    Salva a imagem reconstruída com transformação Log (como na referência).
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dimensões baseadas no modelo
    size_map = {
        "30x30": (30, 30),
        "60x60": (60, 60),
        "large": (360, 360),
        "parse": (60, 60),
    }
    
    img_shape = size_map.get(model_size, (60, 60))
    
    # Converter para array numpy
    img_array = np.array(image_pixels, dtype=np.float64)
    
    # Reshape para 2D
    try:
        if len(img_array) == img_shape[0] * img_shape[1]:
            img_2d = img_array.reshape(img_shape)
        else:
            sqrt_size = int(np.sqrt(len(img_array)))
            if sqrt_size * sqrt_size == len(img_array):
                img_2d = img_array.reshape(sqrt_size, sqrt_size)
                img_shape = (sqrt_size, sqrt_size)
            else:
                return ""
    except Exception:
        return ""
    
    # Aplicar transformação LOG (como mostrado na referência)
    # A referência tem título "Log" - significa escala logarítmica
    img_positive = np.abs(img_2d) + 1e-10
    img_log = np.log10(img_positive)
    
    # Criar figura no estilo da referência
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    
    # Plotar com colormap gray (preto=valores baixos, branco=valores altos)
    im = ax.imshow(img_log, cmap='gray', aspect='equal', origin='upper')
    
    # Eixos com ticks exatamente como na referência
    ax.set_xticks(np.arange(0, img_shape[1]+1, 10))
    ax.set_yticks(np.arange(0, img_shape[0]+1, 10))
    
    # Título "Log" simples como na referência
    ax.set_title("Log", fontsize=14, fontweight='bold')
    
    # Salvar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ultrasound_{server_type}_{model_size}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filepath

# ============================================================
# GERENCIAMENTO DE SERVIDORES
# ============================================================
def compile_cpp() -> bool:
    """Compila o servidor C++ se necessário."""
    exe_path = os.path.join("build", "Release", "UltrasoundBenchmark.exe")
    if not os.path.exists(exe_path):
        console.print("[amarelo]⚙️  Compilando C++...[/amarelo]")
        try:
            subprocess.run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                           check=True, capture_output=True)
            subprocess.run(["cmake", "--build", "build", "--config", "Release"],
                           check=True, capture_output=True)
            console.print("[verde]✅ C++ compilado[/verde]")
            return True
        except:
            console.print("[vermelho]❌ Erro ao compilar C++[/vermelho]")
            return False
    return True

def start_server(server_type: str) -> Optional[subprocess.Popen]:
    """Inicia servidor C++ ou Python."""
    if server_type == "cpp":
        exe_path = os.path.join("build", "Release", "UltrasoundBenchmark.exe")
        if not os.path.exists(exe_path):
            if not compile_cpp():
                return None
        proc = subprocess.Popen([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0)
        port = CPP_PORT
    else:
        proc = subprocess.Popen([sys.executable, "server_python.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0)
        port = PYTHON_PORT
    
    # Aguardar servidor
    for _ in range(60):
        time.sleep(0.5)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((HOST, port))
                return proc
        except:
            pass
    
    proc.terminate()
    return None

def stop_server(proc: subprocess.Popen):
    """Para servidor forçadamente."""
    if proc:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except:
            pass
        try:
            proc.kill()
            proc.wait(timeout=2)
        except:
            pass

# ============================================================
# BENCHMARK PRINCIPAL
# ============================================================
def run_benchmark(
    server_type: str,
    prepared_images: list,
    num_clients: int = 3,
    images_per_client: int = 50,
    log_file: str = None
) -> dict:
    """Executa benchmark completo para um servidor."""
    global logger
    
    port = CPP_PORT if server_type == "cpp" else PYTHON_PORT
    server_name = "C++" if server_type == "cpp" else "Python"
    
    # Header
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold]  🏆 BENCHMARK: SERVIDOR {server_name}[/bold]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"  📊 Clientes: {num_clients}")
    console.print(f"  📷 Imagens por cliente: {images_per_client}")
    console.print(f"  📦 Total: {num_clients * images_per_client} imagens")
    console.print()
    
    if logger:
        logger.log("INICIO", f"Benchmark {server_name}", {
            "clientes": num_clients,
            "imagens_por_cliente": images_per_client
        })
    
    # Iniciar servidor
    console.print(f"[yellow]🚀 Iniciando servidor {server_name}...[/yellow]")
    server_proc = start_server(server_type)
    
    if not server_proc:
        console.print(f"[red]❌ Falha ao iniciar servidor {server_name}[/red]")
        return {}
    
    console.print(f"[green]✅ Servidor {server_name} pronto (porta {port})[/green]")
    
    # Iniciar monitoramento de recursos
    resource_monitor = ResourceMonitor(server_proc.pid, interval=0.5)
    resource_monitor.start()
    
    if logger:
        logger.log("SERVIDOR", f"Servidor {server_name} iniciado", {"porta": port, "pid": server_proc.pid})
    
    # Usar lista de imagens fornecida
    all_images = prepared_images
    
    # Estrutura para resultados
    results_queue = queue.Queue()
    threads = []
    
    # Estatísticas por cliente
    client_stats = {i+1: {
        "latencias": [],
        "solver_times": [],
        "iterations": [],
        "delays": [],
        "imagens_processadas": 0,
        "falhas": 0
    } for i in range(num_clients)}
    
    start_time = time.time()
    
    # Função do cliente
    def thread_client(client_id, port, images, results_q):
        data_dir = "data"
        
        for idx, img_info in enumerate(images):
            # Delay aleatório (requisito do professor)
            delay = random.uniform(0.05, 0.3)
            time.sleep(delay)
            
            signal = load_signal(img_info["file"], data_dir)
            if signal is None:
                results_q.put(("result", {
                    "client_id": client_id, "success": False, "delay_ms": delay * 1000
                }))
                continue
            
            # Aplicar ganho aleatório
            aplicou_ganho = random.choice([True, False])
            if aplicou_ganho:
                signal = apply_gain(signal)
            
            payload = {"model_size": img_info["model"], "signal_g": signal.tolist()}
            
            response, latency_ms = send_request(HOST, port, payload)
            
            result = {
                "client_id": client_id,
                "image_idx": idx + 1,
                "file": img_info["file"],
                "model": img_info["model"],
                "latency_ms": latency_ms,
                "delay_ms": delay * 1000,
                "aplicou_ganho": aplicou_ganho,
                "success": response is not None
            }
            
            if response:
                result["iterations"] = response.get("iterations", 10)
                result["solver_ms"] = response.get("execution_time_ms", 0)
                result["parse_ms"] = response.get("parse_time_ms", 0)
                
                # Salvar imagem se disponível (apenas primeira de cada cliente)
                image_pixels = response.get("image_pixels") or response.get("image")
                if image_pixels and idx == 0:  # Salva apenas a primeira imagem
                    try:
                        saved_path = save_reconstructed_image(
                            image_pixels=image_pixels,
                            model_size=img_info["model"],
                            server_type=server_type,
                            solver_time_ms=result["solver_ms"],
                            iterations=result["iterations"]
                        )
                        result["saved_image"] = saved_path
                    except Exception as e:
                        pass  # Ignorar erros de salvamento
            
            results_q.put(("result", result))
            
            if logger:
                logger.log("IMAGEM", f"Cliente {client_id} - Imagem {idx+1}", result)
        
        results_q.put(("done", client_id))
    
    # Iniciar clientes
    console.print(f"\n[yellow]⚡ Iniciando {num_clients} clientes...[/yellow]")
    for i in range(num_clients):
        t = threading.Thread(target=thread_client, args=(i + 1, port, all_images[i], results_queue))
        t.start()
        threads.append(t)
        console.print(f"  [dim]Cliente {i+1} iniciado[/dim]")
    
    # Progresso por cliente
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Criar barra por cliente
        client_tasks = {}
        for i in range(num_clients):
            client_tasks[i + 1] = progress.add_task(f"[cyan]Cliente {i+1}", total=images_per_client)
        
        clients_done = 0
        
        while clients_done < num_clients:
            try:
                msg_type, data = results_queue.get(timeout=120)
                if msg_type == "result":
                    all_results.append(data)
                    cid = data.get("client_id", 1)
                    
                    if data.get("success"):
                        client_stats[cid]["latencias"].append(data.get("latency_ms", 0))
                        client_stats[cid]["solver_times"].append(data.get("solver_ms", 0))
                        client_stats[cid]["iterations"].append(data.get("iterations", 10))
                        client_stats[cid]["imagens_processadas"] += 1
                    else:
                        client_stats[cid]["falhas"] += 1
                    
                    client_stats[cid]["delays"].append(data.get("delay_ms", 0))
                    
                    # Atualizar barra
                    lats = client_stats[cid]["latencias"]
                    avg = sum(lats) / len(lats) if lats else 0
                    total = client_stats[cid]["imagens_processadas"] + client_stats[cid]["falhas"]
                    
                    progress.update(
                        client_tasks[cid],
                        completed=total,
                        description=f"[cyan]Cliente {cid} | Média: {avg:.0f}ms"
                    )
                elif msg_type == "done":
                    clients_done += 1
            except queue.Empty:
                console.print("[red]Timeout![/red]")
                break
    
    # Aguardar threads
    for t in threads:
        t.join(timeout=5)
    
    total_time = time.time() - start_time
    
    # Parar servidor
    stop_server(server_proc)
    
    # ============================================================
    # ESTATÍSTICAS DETALHADAS
    # ============================================================
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold]  📊 ESTATÍSTICAS DETALHADAS - {server_name}[/bold]")
    console.print(f"[bold green]{'='*70}[/bold green]")
    
    # Tabela por cliente
    table = Table(title="Resultados por Cliente", box=box.ROUNDED)
    table.add_column("Cliente", style="cyan")
    table.add_column("Imagens", style="green")
    table.add_column("Falhas", style="red")
    table.add_column("Latência Média", style="yellow")
    table.add_column("Solver Médio", style="magenta")
    table.add_column("Iterações Médias", style="blue")
    table.add_column("Delay Médio", style="dim")
    
    all_latencies = []
    all_solver = []
    all_iters = []
    
    for cid in range(1, num_clients + 1):
        stats = client_stats[cid]
        lats = stats["latencias"]
        solv = stats["solver_times"]
        iters = stats["iterations"]
        delays = stats["delays"]
        
        all_latencies.extend(lats)
        all_solver.extend(solv)
        all_iters.extend(iters)
        
        table.add_row(
            f"Cliente {cid}",
            str(stats["imagens_processadas"]),
            str(stats["falhas"]),
            f"{sum(lats)/len(lats):.0f}ms" if lats else "N/A",
            f"{sum(solv)/len(solv):.0f}ms" if solv else "N/A",
            f"{sum(iters)/len(iters):.1f}" if iters else "N/A",
            f"{sum(delays)/len(delays):.0f}ms" if delays else "N/A"
        )
    
    console.print(table)
    
    # Estatísticas gerais
    successful = [r for r in all_results if r.get("success")]
    
    stats = {
        "servidor": server_type,
        "nome_servidor": server_name,
        "num_clientes": num_clients,
        "imagens_por_cliente": images_per_client,
        "total_imagens": len(all_results),
        "sucesso": len(successful),
        "falhas": len(all_results) - len(successful),
        "tempo_total_seg": total_time,
        "latencia_media_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
        "latencia_min_ms": min(all_latencies) if all_latencies else 0,
        "latencia_max_ms": max(all_latencies) if all_latencies else 0,
        "solver_medio_ms": sum(all_solver) / len(all_solver) if all_solver else 0,
        "iteracoes_medias": sum(all_iters) / len(all_iters) if all_iters else 0,
        "throughput": len(successful) / total_time if total_time > 0 else 0,
        "resultados_detalhados": all_results,
        "estatisticas_por_cliente": {str(k): v for k, v in client_stats.items()}
    }
    
    # Resumo
    console.print(f"\n[bold]📈 RESUMO {server_name}:[/bold]")
    console.print(f"  ⏱️  Tempo total: {total_time:.2f}s")
    console.print(f"  📷 Imagens processadas: {stats['sucesso']}/{stats['total_imagens']}")
    console.print(f"  ⚡ Throughput: {stats['throughput']:.2f} img/s")
    console.print(f"  📊 Latência média: {stats['latencia_media_ms']:.0f}ms")
    console.print(f"  🔧 Solver médio: {stats['solver_medio_ms']:.0f}ms")
    console.print(f"  🔄 Iterações médias: {stats['iteracoes_medias']:.1f}")
    
    # Parar monitor de recursos e exibir stats
    resource_monitor.stop()
    res_stats = resource_monitor.get_stats()
    
    console.print(f"\n[bold]🖥️  USO DE RECURSOS ({server_name}):[/bold]")
    console.print(f"  🔥 CPU Média: {res_stats['cpu_avg']:.1f}%  |  CPU Máx: {res_stats['cpu_max']:.1f}%")
    console.print(f"  💾 Memória Média: {res_stats['mem_avg']:.1f} MB  |  Memória Máx: {res_stats['mem_max']:.1f} MB")
    console.print(f"  🧵 Threads Média: {res_stats['threads_avg']:.1f}  |  Threads Máx: {int(res_stats['threads_max'])}")
    
    # Adicionar stats de recursos ao resultado
    stats["recursos"] = {
        "cpu_media_pct": res_stats['cpu_avg'],
        "cpu_max_pct": res_stats['cpu_max'],
        "memoria_media_mb": res_stats['mem_avg'],
        "memoria_max_mb": res_stats['mem_max'],
        "threads_media": res_stats['threads_avg'],
        "threads_max": res_stats['threads_max'],
    }
    
    if logger:
        logger.log("RESULTADO", f"Benchmark {server_name} concluído", stats)
    
    return stats


def verificar_justica(cpp_stats: dict, py_stats: dict):
    """Verifica se a batalha foi justa."""
    console.print(f"\n[bold blue]{'='*70}[/bold blue]")
    console.print("[bold]  ⚖️  VERIFICAÇÃO DE JUSTIÇA DA BATALHA[/bold]")
    console.print(f"[bold blue]{'='*70}[/bold blue]")
    
    problemas = []
    
    # Mesmo número de imagens
    if cpp_stats.get("total_imagens") != py_stats.get("total_imagens"):
        problemas.append("❌ Número diferente de imagens processadas")
    else:
        console.print(f"  ✅ Mesmo número de imagens: {cpp_stats.get('total_imagens')}")
    
    # Taxa de sucesso similar
    cpp_taxa = cpp_stats.get("sucesso", 0) / max(cpp_stats.get("total_imagens", 1), 1)
    py_taxa = py_stats.get("sucesso", 0) / max(py_stats.get("total_imagens", 1), 1)
    
    if abs(cpp_taxa - py_taxa) > 0.1:
        problemas.append(f"⚠️  Taxa de sucesso diferente: C++ {cpp_taxa:.0%} vs Python {py_taxa:.0%}")
    else:
        console.print(f"  ✅ Taxa de sucesso similar: C++ {cpp_taxa:.0%} vs Python {py_taxa:.0%}")
    
    # Mesmo número de clientes
    if cpp_stats.get("num_clientes") == py_stats.get("num_clientes"):
        console.print(f"  ✅ Mesmo número de clientes: {cpp_stats.get('num_clientes')}")
    else:
        problemas.append("❌ Número diferente de clientes")
    
    if problemas:
        console.print("\n[yellow]⚠️  PROBLEMAS ENCONTRADOS:[/yellow]")
        for p in problemas:
            console.print(f"  {p}")
    else:
        console.print("\n[green]✅ BATALHA CONSIDERADA JUSTA![/green]")


def comparar_resultados(cpp_stats: dict, python_stats: dict):
    """Compara resultados de C++ e Python."""
    console.print(f"\n[bold magenta]{'='*70}[/bold magenta]")
    console.print(f"[bold]  🏆 COMPARAÇÃO FINAL: C++ vs Python[/bold]")
    console.print(f"[bold magenta]{'='*70}[/bold magenta]")
    
    table = Table(box=box.DOUBLE_EDGE)
    table.add_column("Métrica", style="cyan")
    table.add_column("C++", style="blue")
    table.add_column("Python", style="yellow")
    table.add_column("Diferença", style="green")
    table.add_column("Vencedor", style="bold")
    
    metrics = [
        ("Tempo Total", "tempo_total_seg", "s", True),
        ("Throughput", "throughput", "img/s", False),
        ("Latência Média", "latencia_media_ms", "ms", True),
        ("Tempo Solver", "solver_medio_ms", "ms", True),
        ("Iterações", "iteracoes_medias", "", True),
    ]
    
    # Adicionar métricas de recursos se disponíveis
    cpp_recursos = cpp_stats.get("recursos", {})
    py_recursos = python_stats.get("recursos", {})
    
    if cpp_recursos and py_recursos:
        resource_metrics = [
            ("CPU Média", "cpu_media_pct", "%", True, cpp_recursos, py_recursos),
            ("Memória Máx", "memoria_max_mb", "MB", True, cpp_recursos, py_recursos),
            ("Threads Máx", "threads_max", "", False, cpp_recursos, py_recursos),
        ]
    
    cpp_wins = 0
    py_wins = 0
    
    for name, key, unit, lower_better in metrics:
        cpp_val = cpp_stats.get(key, 0)
        py_val = python_stats.get(key, 0)
        
        if py_val > 0:
            diff = ((cpp_val - py_val) / py_val) * 100
        else:
            diff = 0
        
        if lower_better:
            if cpp_val < py_val:
                winner = "C++ ✅"
                cpp_wins += 1
            else:
                winner = "Python ✅"
                py_wins += 1
        else:
            if cpp_val > py_val:
                winner = "C++ ✅"
                cpp_wins += 1
            else:
                winner = "Python ✅"
                py_wins += 1
        
        table.add_row(
            name,
            f"{cpp_val:.2f}{unit}",
            f"{py_val:.2f}{unit}",
            f"{diff:+.1f}%",
            winner
        )
    
    # Adicionar métricas de recursos à tabela (se disponíveis)
    if cpp_recursos and py_recursos:
        for name, key, unit, lower_better, cpp_res, py_res in resource_metrics:
            cpp_val = cpp_res.get(key, 0)
            py_val = py_res.get(key, 0)
            
            if py_val > 0:
                diff = ((cpp_val - py_val) / py_val) * 100
            else:
                diff = 0
            
            if lower_better:
                winner = "C++ ✅" if cpp_val < py_val else "Python ✅"
            else:
                winner = "C++ ✅" if cpp_val > py_val else "Python ✅"
            
            table.add_row(
                name,
                f"{cpp_val:.1f}{unit}",
                f"{py_val:.1f}{unit}",
                f"{diff:+.1f}%",
                winner
            )
    
    console.print(table)
    
    # Resultado final
    console.print()
    if cpp_wins > py_wins:
        console.print(Panel.fit(
            f"[bold green]🏆 C++ VENCEU! ({cpp_wins} vitórias vs {py_wins})[/bold green]",
            border_style="green"
        ))
    elif py_wins > cpp_wins:
        console.print(Panel.fit(
            f"[bold yellow]🏆 Python VENCEU! ({py_wins} vitórias vs {cpp_wins})[/bold yellow]",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit("[bold blue]🤝 EMPATE![/bold blue]", border_style="blue"))


def main():
    global logger
    
    parser = argparse.ArgumentParser(description="🏆 Benchmark de Reconstrução de Imagem")
    parser.add_argument("--server", choices=["cpp", "python", "both"], default="both")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso")
    parser.add_argument("--cooldown", type=int, default=5, help="Tempo de resfriamento entre execuções")
    parser.add_argument("--model", type=str, default=None, help="Filtrar por modelo (30x30, 60x60, large, parse)")
    args = parser.parse_args()

    # Filtrar sinais
    if args.model:
        SIGNALS[:] = [s for s in SIGNALS if s["model"] == args.model]
        if not SIGNALS:
             console.print(f"[red]❌ Nenhum sinal encontrado para o modelo '{args.model}'[/red]")
             return
    
    # Iniciar logger
    if args.log:
        logger = DetailedLogger(args.log)
        logger.log("CONFIG", "Benchmark iniciado", {
            "servidor": args.server,
            "clientes": args.clients,
            "imagens": args.images
        })
    
    console.print(Panel.fit(
        "[bold cyan]🏆 BENCHMARK DE RECONSTRUÇÃO DE IMAGEM ULTRASSOM[/bold cyan]\n"
        "[dim]Comparativo C++ vs Python - Projeto CGNR[/dim]",
        border_style="cyan"
    ))

    # --- Mostrar Recursos Compartilhados ---
    print(f"\n[bold yellow]🔄 RECURSOS COMPARTILHADOS[/bold yellow]")
    
    # Mostrar Sinais Disponíveis
    print(f"  📂 Sinais Disponíveis ({len(SIGNALS)}):")
    for s in SIGNALS:
        print(f"    - {s['file']} (Modelo: {s['model']})")
    
    # Simular a Fila de Imagens (Deterministicamente)
    print(f"\n  🎯 Pool de Sorteio:")
    print(f"     Os clientes sortearão imagens aleatoriamente deste pool.")
    
    # GERAR FILA DE IMAGENS AGORA (COMPARTILHADA)
    random.seed(42) # Seed fixa para garantir repetibilidade entre rodadas se reiniciar app, mas aqui garante consistencia C++/Py
    shared_images = []
    
    print(f"\n[bold yellow]📜 FILA DE IMAGENS SORTEADA (COMPARTILHADA)[/bold yellow]")
    for i in range(args.clients):
        client_imgs = [random.choice(SIGNALS) for _ in range(args.images)]
        shared_images.append(client_imgs)
        
        # Mostrar resumo da fila para este cliente
        files_counts = {}
        for img in client_imgs:
            fname = img['file']
            files_counts[fname] = files_counts.get(fname, 0) + 1
            
        print(f"  👤 [bold]Cliente {i+1}[/bold] processará {len(client_imgs)} imagens:")
        summary_str = ", ".join([f"{k} (x{v})" for k,v in files_counts.items()])
        print(f"     -> {summary_str}")

    print(f"\n[dim]Todos os servidores (C++/Python) processarão EXATAMENTE esta mesma fila na mesma ordem.[/dim]\n")
    # ---------------------------------------
    
    results = {}
    
    if args.server in ["cpp", "both"]:
        results["cpp"] = run_benchmark("cpp", shared_images, args.clients, args.images, args.log)
    
    if args.server in ["python", "both"]:
        results["python"] = run_benchmark("python", shared_images, args.clients, args.images, args.log)
    
    if "cpp" in results and "python" in results and results["cpp"] and results["python"]:
        verificar_justica(results["cpp"], results["python"])
        comparar_resultados(results["cpp"], results["python"])
    
    # Salvar resultados
    os.makedirs("results", exist_ok=True)
    results_file = f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Remover dados grandes antes de salvar
    for key in results:
        if results[key] and "resultados_detalhados" in results[key]:
            del results[key]["resultados_detalhados"]
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[dim]📁 Resultados: {results_file}[/dim]")
    if args.log:
        console.print(f"[dim]📝 Log detalhado: {args.log}[/dim]")

if __name__ == "__main__":
    main()
