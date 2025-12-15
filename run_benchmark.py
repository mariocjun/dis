#!/usr/bin/env python3
"""
run_benchmark.py - Orchestrator for Ultrasound Benchmark
FIXED: Redirects server logs to files to prevent Pipe Buffer Deadlock on Windows.
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import json
import csv
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import yaml

class SystemMonitor(threading.Thread):
    """Monitors system CPU and RAM usage second by second"""
    def __init__(self, output_file: Path, stop_event: threading.Event):
        super().__init__()
        self.output_file = output_file
        self.stop_event = stop_event
        self.daemon = True

    def run(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_percent', 'memory_percent', 'memory_mb'])
            while not self.stop_event.is_set():
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    writer.writerow([
                        datetime.now().isoformat(),
                        cpu,
                        mem.percent,
                        mem.used / (1024 * 1024)
                    ])
                    f.flush()
                except: pass
                time.sleep(1)

class BenchmarkOrchestrator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / "config.yaml"
        self.effective_config_path: Optional[Path] = None
        self.python_server_process: Optional[subprocess.Popen] = None
        self.cpp_server_process: Optional[subprocess.Popen] = None
        self.output_dir: Optional[Path] = None
        self.py_log_file = None
        self.cpp_log_file = None
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        servers = self.config.get('servers', {})
        self.python_port = servers.get('python', {}).get('port', 5001)
        self.cpp_port = servers.get('cpp', {}).get('port', 5002)
    
    def setup_output_dir(self, suffix: str = "") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"benchmark_{timestamp}"
        if suffix: dir_name += f"_{suffix}"
        
        self.output_dir = self.project_root / "output_runs" / dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "telemetry").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True) # Folder for server logs
        
        self._create_effective_config()
        return self.output_dir
    
    def _create_effective_config(self):
        import copy
        effective_config = copy.deepcopy(self.config)
        effective_config['settings'] = effective_config.get('settings', {}).copy()
        effective_config['settings']['output_base_dir'] = str(self.output_dir.absolute())
        
        if 'datasets' in effective_config:
            for ds in effective_config['datasets']:
                for key in ['h_matrix_csv', 'g_signal_csv']:
                    if key in ds:
                        rel_path = ds[key]
                        abs_path = (self.project_root / rel_path).absolute()
                        ds[key] = str(abs_path)
        
        self.effective_config_path = self.output_dir / "config_effective.yaml"
        with open(self.effective_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(effective_config, f, default_flow_style=False, allow_unicode=True)

    def start_python_server(self) -> bool:
        print("[INFO] Starting Python server...")
        server_script = self.project_root / "server" / "python_server.py"
        
        # Redirect output to file to avoid buffer blocking
        self.py_log_file = open(self.output_dir / "logs" / "server_python.log", "w")
        
        self.python_server_process = subprocess.Popen(
            [sys.executable, str(server_script), "--config", str(self.effective_config_path), "--port", str(self.python_port)],
            cwd=str(self.project_root), 
            stdout=self.py_log_file, 
            stderr=subprocess.STDOUT
        )
        time.sleep(2)
        if self.python_server_process.poll() is not None:
            print("[ERROR] Python server failed to start. Check logs/server_python.log")
            return False
        return True
    
    def start_cpp_server(self) -> bool:
        print("[INFO] Starting C++ server...")
        exe_name = "UltrasoundServerHTTP.exe" if sys.platform == "win32" else "UltrasoundServerHTTP"
        exe_path = self.project_root / "build" / exe_name
        if not exe_path.exists(): exe_path = self.project_root / "build" / "Release" / exe_name
        
        if not exe_path.exists():
            print(f"[ERROR] C++ executable not found at {exe_path}")
            return False
            
        # Redirect output to file
        self.cpp_log_file = open(self.output_dir / "logs" / "server_cpp.log", "w")

        self.cpp_server_process = subprocess.Popen(
            [str(exe_path), "--config", str(self.effective_config_path), "--port", str(self.cpp_port)],
            cwd=str(self.project_root), 
            stdout=self.cpp_log_file, 
            stderr=subprocess.STDOUT
        )
        time.sleep(2)
        if self.cpp_server_process.poll() is not None:
            print("[ERROR] C++ server failed to start. Check logs/server_cpp.log")
            return False
        return True
    
    def stop_servers(self):
        print("[INFO] Stopping servers...")
        for p in [self.python_server_process, self.cpp_server_process]:
            if p:
                p.terminate()
                try: p.wait(timeout=5)
                except: p.kill()
        
        # Close log files
        if self.py_log_file: self.py_log_file.close()
        if self.cpp_log_file: self.cpp_log_file.close()

    def run_3_clients(self, num_jobs_per_client: int, seed_base: int, datasets: List[str], python_only: bool, cpp_only: bool):
        client_script = self.project_root / "client" / "client_generator.py"
        
        stop_monitor = threading.Event()
        monitor_csv = self.output_dir / "telemetry" / "system_metrics.csv"
        monitor = SystemMonitor(monitor_csv, stop_monitor)
        monitor.start()
        print(f"[INFO] System Monitor started.")

        servers_to_test = []
        if not cpp_only: servers_to_test.append(('python', self.python_port))
        if not python_only: servers_to_test.append(('cpp', self.cpp_port))

        for server_name, port in servers_to_test:
            print(f"\n{'='*60}")
            print(f"LAUNCHING 3 CLIENTS AGAINST {server_name.upper()} SERVER")
            print(f"{'='*60}")
            
            client_procs = []
            for i in range(3):
                client_seed = seed_base + i
                cmd = [
                    sys.executable, str(client_script),
                    "--seed", str(client_seed),
                    "--num-jobs", str(num_jobs_per_client),
                    "--datasets"
                ] + datasets
                
                if server_name == 'python':
                    cmd.extend(["--python-only", "--python-url", f"http://localhost:{port}"])
                else:
                    cmd.extend(["--cpp-only", "--cpp-url", f"http://localhost:{port}"])
                
                cmd.extend(["--output-dir", str(self.output_dir)])

                # Clients inherit stdout so we can see their progress in console
                p = subprocess.Popen(cmd)
                client_procs.append(p)
            
            for p in client_procs:
                p.wait()
            print(f"[ORCHESTRATOR] All 3 clients finished for {server_name}.")

        stop_monitor.set()
        monitor.join()
        print("[INFO] System Monitor stopped.")

    def generate_report(self):
        print("\n[INFO] Generating report...")
        report_script = self.project_root / "scripts" / "generate_report.py"
        subprocess.run([sys.executable, str(report_script), "--input-dir", str(self.output_dir)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-jobs', type=int, default=5)
    parser.add_argument('--python-only', action='store_true')
    parser.add_argument('--cpp-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', default=['30x30_g1', '60x60_G1'])
    args = parser.parse_args()
    
    orchestrator = BenchmarkOrchestrator(Path(__file__).parent.absolute())
    
    def cleanup(signum, frame):
        orchestrator.stop_servers()
        sys.exit(1)
    signal.signal(signal.SIGINT, cleanup)
    
    try:
        orchestrator.setup_output_dir(f"seed{args.seed}")
        
        if not args.cpp_only: orchestrator.start_python_server()
        if not args.python_only: orchestrator.start_cpp_server()
        
        time.sleep(2)
        
        orchestrator.run_3_clients(
            num_jobs_per_client=args.num_jobs,
            seed_base=args.seed,
            datasets=args.datasets,
            python_only=args.python_only,
            cpp_only=args.cpp_only
        )
        
        orchestrator.generate_report()
        print(f"\n[SUCCESS] Benchmark Complete. Results in {orchestrator.output_dir}")
        
    finally:
        orchestrator.stop_servers()

if __name__ == '__main__':
    main()