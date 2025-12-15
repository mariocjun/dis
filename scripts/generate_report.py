#!/usr/bin/env python3
"""
generate_report.py - Updated with Robust Image Generation (Fixes White Images)
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
        ax1.plot(df['elapsed'], df['cpu_percent'], color=color, label='CPU %', linewidth=1)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Memory Usage (MB)', color=color)
        ax2.plot(df['elapsed'], df['memory_mb'], color=color, label='RAM (MB)', linewidth=1)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('System Resource Usage During Benchmark')
        fig.tight_layout()
        
        output_path = output_dir / 'graphs' / 'system_resources.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    except Exception as e:
        print(f"[WARN] Failed to plot system resources: {e}")
        return None

def generate_robust_png(csv_path: Path, output_dir: Path):
    """Generates PNG handling NaNs, Infs and Contrast issues"""
    if not HAS_MATPLOTLIB: return None
    
    try:
        # Load Data
        data = np.loadtxt(csv_path, delimiter=',')
        
        # 1. Handle Explosion (NaN/Inf)
        if not np.isfinite(data).all():
            print(f"[WARN] Image {csv_path.name} contains NaN/Inf. Fixing for display.")
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            
        # 2. Handle Contrast (Use Absolute value for CGNR and Clip outliers)
        # CGNR represents signal intensity, usually positive. 
        # Negative values are often artifacts. We take abs().
        data_abs = np.abs(data)
        
        # Clip top 1% of brightness to avoid one hot pixel hiding everything
        v_min = 0
        v_max = np.percentile(data_abs, 99.5) 
        if v_max == 0: v_max = data_abs.max() # Fallback if empty
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(data_abs, cmap='inferno', vmin=v_min, vmax=v_max, aspect='equal')
        plt.colorbar(im, ax=ax, label='Signal Intensity (Abs)')
        
        # Add Metadata Title
        meta_path = csv_path.parent / csv_path.name.replace('_image.csv', '_meta.json')
        title = csv_path.stem
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            err = meta.get('final_error', 0)
            # Mark as DIVERGED in title if error is huge
            status = "DIVERGED" if err > 1e5 else "Converged"
            title = f"{meta.get('dataset_id')} | {status}\nErr: {err:.2e} | Iter: {meta.get('iterations')}"
            
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        png_path = output_dir / 'images' / csv_path.name.replace('.csv', '.png')
        plt.savefig(png_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        return png_path
    except Exception as e:
        print(f"[WARN] Failed to generate PNG for {csv_path.name}: {e}")
        return None

def generate_report(input_dir: Path, output_dir: Path):
    all_data = load_all_data(input_dir)
    
    # Generate System Graph
    sys_graph = generate_system_resource_graph(input_dir, output_dir)
    
    # Generate All Images (Robustly)
    images_dir = input_dir / 'images'
    generated_images = []
    if images_dir.exists():
        for csv_file in images_dir.glob("*_image.csv"):
            png = generate_robust_png(csv_file, output_dir)
            if png: generated_images.append(png)
    
    # Stats
    py_jobs = [d for d in all_data if d['server'] == 'python']
    cpp_jobs = [d for d in all_data if d['server'] == 'cpp']
    
    report_path = output_dir / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Report (Final)\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("## System Resources\n")
        if sys_graph: f.write(f"![Resources](graphs/system_resources.png)\n\n")
        
        f.write("## Performance Summary\n")
        if py_jobs and cpp_jobs:
            try:
                py_time = np.mean([float(j['solver_time_ms']) for j in py_jobs])
                cpp_time = np.mean([float(j['solver_time_ms']) for j in cpp_jobs])
                speedup = py_time / cpp_time if cpp_time > 0 else 0
                f.write(f"- **Speedup: {speedup:.2f}x** (C++ vs Python)\n")
                f.write(f"- Avg Python Time: {py_time:.2f} ms\n")
                f.write(f"- Avg C++ Time: {cpp_time:.2f} ms\n")
            except: pass

        f.write("\n## Reconstructed Images (Sample)\n")
        # Show a few images (mix of converged and diverged if any)
        count = 0
        for img in sorted(generated_images):
            if count >= 12: break # Limit to 12 images in report
            f.write(f"![Image](images/{img.name})\n")
            count += 1

    print(f"[INFO] Report regenerated at {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=False)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else Path(args.input_dir)
    generate_report(Path(args.input_dir), out)