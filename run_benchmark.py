#!/usr/bin/env python3
"""
run_benchmark.py - Scientific Orchestrator for Ultrasound Reconstruction
Mode: DEMO (Deterministic) & STRESS (Random)
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
import platform
import shutil
from datetime import datetime
from pathlib import Path
import yaml

# --- System Monitor Thread ---
class SystemMonitor(threading.Thread):
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

# --- Main Orchestrator ---
class BenchmarkOrchestrator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / "config.yaml"
        self.output_dir: Optional[Path] = None
        self.py_log_file = None
        self.cpp_log_file = None
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.python_port = self.config['servers']['python']['port']
        self.cpp_port = self.config['servers']['cpp']['port']

    def get_environment_snapshot(self) -> dict:
        """Captures hardware/software environment for reproducibility"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version.split()[0],
            "compiler_info": "Unknown (Check build logs)"
        }
        # Try to get C++ compiler info
        try:
            res = subprocess.run(["g++", "--version"], capture_output=True, text=True)
            if res.returncode == 0: info["compiler_info"] = res.stdout.split('\n')[0]
        except: pass
        return info

    def setup_experiment(self, mode: str) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f"{timestamp}_{mode.upper()}"
        self.output_dir = self.project_root / "execs" / dir_name
        
        # Create structure
        for sub in ["images", "telemetry", "graphs", "logs", "data"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
            
        # Save Environment Snapshot
        env_info = self.get_environment_snapshot()
        with open(self.output_dir / "data" / "environment.json", "w") as f:
            json.dump(env_info, f, indent=2)
            
        # Save Config Snapshot
        self._create_effective_config()
        
        print(f"\n[SETUP] Experiment initialized at: {self.output_dir}")
        print(f"[SETUP] Environment: {env_info['platform']} | {env_info['cpu_cores_logical']} vCPUs | {env_info['ram_total_gb']} GB RAM")
        return self.output_dir

    def _create_effective_config(self):
        import copy
        effective_config = copy.deepcopy(self.config)
        effective_config['settings']['output_base_dir'] = str(self.output_dir.absolute())
        
        # Absolute paths for data
        if 'datasets' in effective_config:
            for ds in effective_config['datasets']:
                for key in ['h_matrix_csv', 'g_signal_csv']:
                    if key in ds:
                        ds[key] = str((self.project_root / ds[key]).absolute())
        
        self.effective_config_path = self.output_dir / "data" / "config_snapshot.yaml"
        with open(self.effective_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(effective_config, f)

    def start_servers(self):
        print("[INFO] Starting Servers...")
        
        # Python
        self.py_log_file = open(self.output_dir / "logs" / "server_python.log", "w")
        self.py_proc = subprocess.Popen(
            [sys.executable, "server/python_server.py", "--config", str(self.effective_config_path), "--port", str(self.python_port)],
            cwd=str(self.project_root), stdout=self.py_log_file, stderr=subprocess.STDOUT
        )
        
        # C++
        exe = "UltrasoundServerHTTP.exe" if sys.platform == "win32" else "UltrasoundServerHTTP"
        exe_path = self.project_root / "build" / exe
        if not exe_path.exists(): exe_path = self.project_root / "build" / "Release" / exe
        
        self.cpp_log_file = open(self.output_dir / "logs" / "server_cpp.log", "w")
        self.cpp_proc = subprocess.Popen(
            [str(exe_path), "--config", str(self.effective_config_path), "--port", str(self.cpp_port)],
            cwd=str(self.project_root), stdout=self.cpp_log_file, stderr=subprocess.STDOUT
        )
        time.sleep(2) # Warmup

    def stop_servers(self):
        print("[INFO] Stopping Servers...")
        for p in [self.py_proc, self.cpp_proc]:
            if p: p.terminate()
        if self.py_log_file: self.py_log_file.close()
        if self.cpp_log_file: self.cpp_log_file.close()

    def run_client_job(self, job_prefix, num_jobs, datasets, concurrency=1, target_server=None):
        """Generic client runner"""
        client_script = self.project_root / "client" / "client_generator.py"
        procs = []
        
        targets = []
        if target_server == 'python' or target_server is None: targets.append(('python', self.python_port))
        if target_server == 'cpp' or target_server is None: targets.append(('cpp', self.cpp_port))

        # Demo mode prefixes use fixed gain=1.0, manual mode uses random gain
        is_demo_job = job_prefix in ["sanity", "race_30", "race_60", "sat"]
        
        for srv_name, port in targets:
            for i in range(concurrency):
                cmd = [
                    sys.executable, str(client_script),
                    "--seed", str(42 + i), # Deterministic seeds
                    "--num-jobs", str(num_jobs),
                    "--output-dir", str(self.output_dir),
                ]
                
                # Fixed gain=1.0 for demo mode, random for manual
                if is_demo_job:
                    cmd.extend(["--gain-min", "1.0", "--gain-max", "1.0"])
                
                cmd.extend(["--datasets"] + datasets)
                
                if srv_name == 'python': cmd.extend(["--python-only", "--python-url", f"http://localhost:{port}"])
                else: cmd.extend(["--cpp-only", "--cpp-url", f"http://localhost:{port}"])
                
                # Add prefix to identify phase in logs (requires client update or just rely on timestamp/order)
                # For now, we rely on the sequential execution of this script.
                
                p = subprocess.Popen(cmd)
                procs.append(p)
        
        for p in procs: p.wait()

    def run_demo_mode(self):
        """
        Executes the Scientific Demo Protocol:
        1. Sanity Check (1 job)
        2. The Race (3 reps for Std Dev)
        3. Saturation (3 concurrent clients)
        """
        print("\n" + "="*60)
        print("🚀 STARTING SCIENTIFIC DEMO PROTOCOL")
        print("="*60)
        
        # All available datasets for comprehensive benchmark
        all_30x30 = ["30x30_g1", "30x30_g2", "30x30_A1"]
        all_60x60 = ["60x60_G1", "60x60_G2", "60x60_A1"]
        all_datasets = all_30x30 + all_60x60

        # --- ACT 1: SANITY CHECK ---
        print("\n>>> ACT 1: SANITY CHECK (Warmup)")
        self.run_client_job("sanity", 1, ["30x30_g1"], concurrency=1)

        # --- ACT 2: THE RACE (Variability Analysis) ---
        print("\n>>> ACT 2: THE RACE (Performance & Variability)")
        # Run 3 times for each size to calculate Std Dev
        print("    Testing 30x30 datasets (3 repetitions each)...")
        self.run_client_job("race_30", 3, all_30x30, concurrency=1)
        print("    Testing 60x60 datasets (3 repetitions each)...")
        self.run_client_job("race_60", 3, all_60x60, concurrency=1)

        # --- ACT 3: SATURATION ---
        print("\n>>> ACT 3: SATURATION (Stress Test)")
        stop_monitor = threading.Event()
        monitor = SystemMonitor(self.output_dir / "telemetry" / "system_metrics.csv", stop_monitor)
        monitor.start()
        
        # 3 Concurrent clients, all datasets
        self.run_client_job("sat", 5, all_datasets, concurrency=3)
        
        stop_monitor.set()
        monitor.join()

    def run_full_mode(self, num_reps=10, concurrency=4):
        """
        Full benchmark mode with:
        - All datasets (30x30 and 60x60)
        - Random gains (0.5 to 2.0) with same seed for C++ and Python
        - Multiple repetitions for statistical significance
        - Stress test with high concurrency
        """
        print("\n" + "="*60)
        print("🔬 STARTING FULL BENCHMARK MODE")
        print(f"   Repetitions: {num_reps} per dataset")
        print(f"   Concurrency: {concurrency} clients")
        print(f"   Gain: Random (0.5 - 2.0)")
        print("="*60)
        
        all_30x30 = ["30x30_g1", "30x30_g2", "30x30_A1"]
        all_60x60 = ["60x60_G1", "60x60_G2", "60x60_A1"]
        all_datasets = all_30x30 + all_60x60

        # --- WARMUP ---
        print("\n>>> PHASE 1: WARMUP")
        self.run_full_job("warmup", 1, ["30x30_g1"], concurrency=1)

        # --- 30x30 TESTS ---
        print(f"\n>>> PHASE 2: 30x30 DATASETS ({num_reps} reps each, random gains)")
        self.run_full_job("full_30", num_reps, all_30x30, concurrency=1)

        # --- 60x60 TESTS ---
        print(f"\n>>> PHASE 3: 60x60 DATASETS ({num_reps} reps each, random gains)")
        self.run_full_job("full_60", num_reps, all_60x60, concurrency=1)

        # --- STRESS TEST ---
        print(f"\n>>> PHASE 4: STRESS TEST ({concurrency} concurrent clients)")
        stop_monitor = threading.Event()
        monitor = SystemMonitor(self.output_dir / "telemetry" / "system_metrics.csv", stop_monitor)
        monitor.start()
        
        self.run_full_job("stress", num_reps, all_datasets, concurrency=concurrency)
        
        stop_monitor.set()
        monitor.join()

    def run_full_job(self, job_prefix, num_jobs, datasets, concurrency=1, target_server=None):
        """Run jobs with random gains (0.5-2.0) using same seed for both servers."""
        client_script = self.project_root / "scripts" / "client_generator.py"
        procs = []
        
        targets = []
        if target_server == 'python' or target_server is None: targets.append(('python', self.python_port))
        if target_server == 'cpp' or target_server is None: targets.append(('cpp', self.cpp_port))
        
        for srv_name, port in targets:
            for i in range(concurrency):
                cmd = [
                    sys.executable, str(client_script),
                    "--seed", str(42 + i),  # Same seed for both servers
                    "--num-jobs", str(num_jobs),
                    "--output-dir", str(self.output_dir),
                    "--gain-min", "0.5",
                    "--gain-max", "2.0",  # Random gains
                ]
                
                cmd.extend(["--datasets"] + datasets)
                
                if srv_name == 'python': 
                    cmd.extend(["--python-only", "--python-url", f"http://localhost:{port}"])
                else: 
                    cmd.extend(["--cpp-only", "--cpp-url", f"http://localhost:{port}"])
                
                p = subprocess.Popen(cmd)
                procs.append(p)
        
        for p in procs: p.wait()


    def generate_report(self):
        print("\n[INFO] Generating Scientific Report (HTML)...")
        # Priority: HTML > Markdown > Legacy
        report_script = self.project_root / "scripts" / "generate_report_html.py"
        if not report_script.exists():
            report_script = self.project_root / "scripts" / "generate_report_md.py"
        if not report_script.exists():
            report_script = self.project_root / "scripts" / "generate_report.py"
        subprocess.run([sys.executable, str(report_script), "--input-dir", str(self.output_dir)])

def main():
    parser = argparse.ArgumentParser(description="Ultrasound Reconstruction Benchmark")
    parser.add_argument('--demo', action='store_true', help="Run deterministic scientific demo (fixed gain=1.0)")
    parser.add_argument('--full', action='store_true', help="Run FULL benchmark: all datasets, random gains, multiple reps")
    parser.add_argument('--reps', type=int, default=10, help="Number of repetitions per dataset (default: 10)")
    parser.add_argument('--concurrency', type=int, default=4, help="Max concurrent clients for stress test (default: 4)")
    args = parser.parse_args()
    
    # Determine mode name
    if args.full:
        mode_name = f"FULL_r{args.reps}_c{args.concurrency}"
    elif args.demo:
        mode_name = "DEMO"
    else:
        mode_name = "MANUAL"
    
    orch = BenchmarkOrchestrator(Path(__file__).parent.absolute())
    
    def cleanup(signum, frame):
        orch.stop_servers()
        sys.exit(1)
    signal.signal(signal.SIGINT, cleanup)
    
    try:
        orch.setup_experiment(mode_name)
        orch.start_servers()
        
        if args.full:
            orch.run_full_mode(num_reps=args.reps, concurrency=args.concurrency)
        elif args.demo:
            orch.run_demo_mode()
        else:
            # Default manual run (backward compatibility)
            orch.run_client_job("manual", 5, ["30x30_g1", "60x60_G1"], concurrency=3)
            
        orch.generate_report()
        print(f"\n[SUCCESS] Experiment Complete. Report at: {orch.output_dir}/Relatorio_Cientifico.html")
        
    finally:
        orch.stop_servers()

if __name__ == '__main__':
    main()