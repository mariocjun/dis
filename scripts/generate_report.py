#!/usr/bin/env python3
"""
generate_report.py - Updated to include System Resource Graphs
"""

import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib/pandas not available, skipping graphs")

def load_all_data(input_dir: Path):
    all_data = []
    job_metrics_path = input_dir / 'telemetry' / 'job_metrics.csv'
    if job_metrics_path.exists():
        with open(job_metrics_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_data.extend(list(reader))
    return all_data

def generate_system_resource_graph(input_dir: Path, output_dir: Path):
    """Generates CPU and Memory usage over time graph"""
    if not HAS_MATPLOTLIB: return None
    
    csv_path = input_dir / 'telemetry' / 'system_metrics.csv'
    if not csv_path.exists(): return None

    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        start_time = df['timestamp'].min()
        df['elapsed'] = (df['timestamp'] - start_time).dt.total_seconds()

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:red'
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('CPU Usage (%)', color=color)
        ax1.plot(df['elapsed'], df['cpu_percent'], color=color, label='CPU %')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Memory Usage (MB)', color=color)
        ax2.plot(df['elapsed'], df['memory_mb'], color=color, label='RAM (MB)')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('System Resource Usage During Benchmark (3 Clients)')
        fig.tight_layout()
        
        output_path = output_dir / 'graphs' / 'system_resources.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    except Exception as e:
        print(f"[WARN] Failed to plot system resources: {e}")
        return None

def generate_report(input_dir: Path, output_dir: Path):
    all_data = load_all_data(input_dir)
    
    # Generate Graphs
    sys_graph = generate_system_resource_graph(input_dir, output_dir)
    
    py_jobs = [d for d in all_data if d['server'] == 'python']
    cpp_jobs = [d for d in all_data if d['server'] == 'cpp']
    
    report_path = output_dir / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Report (3 Clients)\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("## System Resources (Second by Second)\n")
        if sys_graph:
            f.write(f"![System Resources](graphs/system_resources.png)\n\n")
        else:
            f.write("Graph not available (check if pandas/matplotlib are installed).\n\n")
            
        f.write("## Job Summary\n")
        f.write(f"- Python Jobs Completed: {len(py_jobs)}\n")
        f.write(f"- C++ Jobs Completed: {len(cpp_jobs)}\n")
        
        if py_jobs and cpp_jobs:
            try:
                py_time = np.mean([float(j['solver_time_ms']) for j in py_jobs])
                cpp_time = np.mean([float(j['solver_time_ms']) for j in cpp_jobs])
                speedup = py_time / cpp_time if cpp_time > 0 else 0
                f.write(f"\n## Performance\n")
                f.write(f"- Avg Python Solver Time: {py_time:.2f} ms\n")
                f.write(f"- Avg C++ Solver Time: {cpp_time:.2f} ms\n")
                f.write(f"- **Speedup: {speedup:.2f}x**\n")
            except:
                pass

    print(f"[INFO] Report generated at {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=False)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else Path(args.input_dir)
    generate_report(Path(args.input_dir), out)