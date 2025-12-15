#!/usr/bin/env python3
"""
generate_report_html.py - Beautiful HTML Scientific Report Generator
Creates a modern, responsive HTML report with CSS styling and animations.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import yaml
import glob
import imageio


def get_config_values(input_dir: Path) -> dict:
    """Get values from config file."""
    config_path = input_dir / 'data' / 'config_snapshot.yaml'
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            pass
    return {}


def format_error(value: float) -> str:
    """Format error in scientific notation for readability."""
    if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.2e}"
    return f"{value:.4f}"


def create_iteration_frames(img_dir: Path, output_dir: Path, job_id: str, server: str) -> list:
    """Create individual PNG frames from iteration images for carousel."""
    # Look for iteration files: cpp_jobid_iter_N.csv or jobid_iter_N.csv
    prefix = "cpp_" if server.lower() == 'cpp' else ""
    pattern = f"{prefix}{job_id}_iter_*.csv"
    iter_files = glob.glob(str(img_dir / pattern))
    
    # Sort numerically by iteration number
    def get_iter_num(f):
        try:
            return int(Path(f).stem.split('_iter_')[-1])
        except:
            return 0
    iter_files = sorted(iter_files, key=get_iter_num)
    
    if len(iter_files) < 2:
        return []
    
    frame_paths = []
    
    for i, iter_file in enumerate(iter_files):
        try:
            data = np.loadtxt(iter_file, delimiter=',')
            
            # Same settings as main image
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            # Same transform as main image
            data_rotated = np.flipud(np.rot90(data, k=1))
            
            v_min, v_max = np.min(data_rotated), np.max(data_rotated)
            if v_max <= v_min:
                v_max = v_min + 1.0
            
            im = ax.imshow(data_rotated, cmap='inferno', vmin=v_min, vmax=v_max,
                           aspect='equal', interpolation='nearest')
            
            # Colorbar with white text (same as main)
            cbar = fig.colorbar(im, ax=ax, label='Intensidade')
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.label.set_color('white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
            
            ax.axis('off')
            
            frame_name = f"iter_{job_id}_{server}_{i:02d}.png"
            frame_path = output_dir / frame_name
            fig.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            plt.close(fig)
            
            frame_paths.append(frame_name)
        except Exception as e:
            print(f"[WARN] Failed to create frame {i}: {e}")
    
    return frame_paths


def create_image_png(csv_path: Path, output_path: Path, dataset: str, error: float, threshold: float):
    """Create PNG from CSV image data with conditional rotation/flip."""
    data = np.loadtxt(csv_path, delimiter=',')
    
    # Rotate 90 degrees counter-clockwise (left) and flip vertically
    data_rotated = np.flipud(np.rot90(data, k=1))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    v_min, v_max = np.min(data_rotated), np.max(data_rotated)
    if v_max <= v_min:
        v_max = v_min + 1.0
    
    im = ax.imshow(data_rotated, cmap='inferno', vmin=v_min, vmax=v_max,
                   aspect='equal', interpolation='nearest')
    
    # Colorbar with white text
    cbar = fig.colorbar(im, ax=ax, label='Intensidade')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    
    ax.axis('off')
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    return output_path


def create_lcurve_plot(img_dir: Path, output_dir: Path, job_id: str, server: str, dataset: str) -> str:
    """Create L-curve plot. Prefers CSV L-curve data, falls back to iteration calculation.
    
    Traditional L-curve: X = ||x|| (solution norm), Y = ||r|| (residual norm).
    """
    prefix = "cpp_" if server.lower() == 'cpp' else ""
    
    # First try to find L-curve CSV file
    lcurve_csv = img_dir / f"{job_id}_lcurve.csv"
    cpp_lcurve_csv = img_dir / f"cpp_{job_id}_lcurve.csv"
    
    solution_norms = []
    residual_norms = []
    
    # Try to load from L-curve CSV (preferred - has real data)
    target_csv = cpp_lcurve_csv if server.lower() == 'cpp' else lcurve_csv
    if target_csv.exists():
        try:
            df = pd.read_csv(target_csv)
            solution_norms = df['SolutionNorm'].tolist()
            residual_norms = df['ResidualNorm'].tolist()
        except Exception as e:
            print(f"[WARN] Failed to read L-curve CSV: {e}")
    
    # Fallback: calculate from iteration images if no CSV
    if not solution_norms:
        pattern = f"{prefix}{job_id}_iter_*.csv"
        iter_files = glob.glob(str(img_dir / pattern))
        
        if len(iter_files) < 2:
            return ""
        
        def get_iter_num(f):
            try:
                return int(Path(f).stem.split('_iter_')[-1])
            except:
                return 0
        iter_files = sorted(iter_files, key=get_iter_num)
        
        for i, iter_file in enumerate(iter_files):
            try:
                data = np.loadtxt(iter_file, delimiter=',')
                sol_norm = np.linalg.norm(data.flatten())
                if sol_norm > 0:
                    solution_norms.append(sol_norm)
                    # Use change in norm as residual proxy (fallback)
                    if len(residual_norms) == 0:
                        residual_norms.append(sol_norm)
                    else:
                        residual_norms.append(abs(sol_norm - solution_norms[-2]))
            except:
                continue
    
    if len(solution_norms) < 2 or len(residual_norms) < 2:
        return ""
    
    # Create L-curve plot (traditional: solution norm vs residual norm, log-log)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    color = '#f97316' if server.lower() == 'cpp' else '#3b82f6'
    
    # Filter out zeros for log scale
    valid_points = [(x, y) for x, y in zip(solution_norms, residual_norms) if x > 0 and y > 0]
    if len(valid_points) < 2:
        plt.close(fig)
        return ""
    
    x_vals, y_vals = zip(*valid_points)
    
    # Plot with log-log scale
    ax.loglog(x_vals, y_vals, 'o-', color=color, linewidth=2, markersize=8)
    
    # Mark elbow point (middle of curve as approximation)
    if len(x_vals) >= 3:
        elbow_idx = len(x_vals) // 2
        ax.plot(x_vals[elbow_idx], y_vals[elbow_idx], 's', 
               color='#22c55e', markersize=14, label=f'Cotovelo (iter {elbow_idx+1})', zorder=5)
    
    ax.set_xlabel('||x|| (Norma da Solução)', color='white', fontsize=12)
    ax.set_ylabel('||r|| (Norma do Resíduo)', color='white', fontsize=12)
    ax.set_title(f'Curva L - {dataset} ({server.upper()})', color='white', fontsize=14)
    ax.tick_params(axis='both', which='both', colors='white', labelcolor='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend(facecolor='#252542', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='white')
    
    filename = f"lcurve_{dataset}_{server}.png"
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    return filename


def create_resource_usage_charts(input_dir: Path, output_dir: Path) -> dict:
    """Create CPU and Memory usage timeline charts. Returns dict with filenames."""
    metrics_path = input_dir / 'telemetry' / 'system_metrics.csv'
    if not metrics_path.exists():
        return {}
    
    try:
        df = pd.read_csv(metrics_path)
    except:
        return {}
    
    if df.empty:
        return {}
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['elapsed_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    result = {}
    
    # CPU Usage Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    ax.fill_between(df['elapsed_sec'], df['cpu_percent'], alpha=0.3, color='#7c3aed')
    ax.plot(df['elapsed_sec'], df['cpu_percent'], color='#a78bfa', linewidth=2)
    
    ax.set_xlabel('Tempo (segundos)', color='white', fontsize=12)
    ax.set_ylabel('Uso de CPU (%)', color='white', fontsize=12)
    ax.set_title('Uso de CPU Durante Teste de Saturação', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_ylim(0, 100)
    ax.axhline(y=df['cpu_percent'].mean(), color='#ef4444', linestyle='--', label=f"Média: {df['cpu_percent'].mean():.1f}%")
    ax.legend(facecolor='#252542', edgecolor='white', labelcolor='white')
    
    cpu_file = "cpu_usage.png"
    fig.savefig(output_dir / cpu_file, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    result['cpu'] = cpu_file
    
    # Memory Usage Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    memory_gb = df['memory_mb'] / 1024
    ax.fill_between(df['elapsed_sec'], memory_gb, alpha=0.3, color='#22c55e')
    ax.plot(df['elapsed_sec'], memory_gb, color='#4ade80', linewidth=2)
    
    ax.set_xlabel('Tempo (segundos)', color='white', fontsize=12)
    ax.set_ylabel('Memória Usada (GB)', color='white', fontsize=12)
    ax.set_title('Uso de Memória Durante Teste de Saturação', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.axhline(y=memory_gb.mean(), color='#ef4444', linestyle='--', label=f"Média: {memory_gb.mean():.1f} GB")
    ax.legend(facecolor='#252542', edgecolor='white', labelcolor='white')
    
    mem_file = "memory_usage.png"
    fig.savefig(output_dir / mem_file, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    result['memory'] = mem_file
    
    return result


def create_per_job_memory_chart(df: pd.DataFrame, output_dir: Path) -> str:
    """Create bar chart showing memory usage per job by server."""
    # Only use jobs that have RAM data (Python jobs typically have it)
    df_with_ram = df[df['ram_peak_mb'] > 0].copy()
    
    if df_with_ram.empty:
        return ""
    
    # Group by dataset and server, get average RAM
    grouped = df_with_ram.groupby(['dataset_id', 'server']).agg({
        'ram_peak_mb': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    datasets = sorted(grouped['dataset_id'].unique())
    x = np.arange(len(datasets))
    width = 0.35
    
    cpp_data = grouped[grouped['server'] == 'cpp'].set_index('dataset_id').reindex(datasets)
    py_data = grouped[grouped['server'] == 'python'].set_index('dataset_id').reindex(datasets)
    
    if not cpp_data.empty and cpp_data['ram_peak_mb'].notna().any():
        ax.bar(x - width/2, cpp_data['ram_peak_mb'].fillna(0), width, 
              label='C++', color='#f97316')
    
    if not py_data.empty and py_data['ram_peak_mb'].notna().any():
        ax.bar(x + width/2, py_data['ram_peak_mb'].fillna(0), width,
              label='Python', color='#3b82f6')
    
    ax.set_xlabel('Dataset', color='white', fontsize=12)
    ax.set_ylabel('Memória Pico (MB)', color='white', fontsize=12)
    ax.set_title('Uso de Memória por Job (RAM Pico)', color='white', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend(facecolor='#252542', edgecolor='white', labelcolor='white')
    ax.grid(axis='y', alpha=0.3, color='white')
    
    plt.tight_layout()
    filename = "memory_per_job.png"
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    return filename


def create_race_comparison_charts(df: pd.DataFrame, output_dir: Path) -> dict:
    """Create race comparison bar charts for 30x30 and 60x60 datasets. Returns dict with filenames."""
    result = {}
    
    for size_label, size_pattern in [('30x30', '30x30'), ('60x60', '60x60')]:
        # Filter by size
        size_df = df[df['dataset_id'].str.contains(size_pattern, case=False)]
        if size_df.empty:
            continue
        
        # Group by server and dataset
        grouped = size_df.groupby(['dataset_id', 'server']).agg({
            'solver_time_ms': ['mean', 'std'],
            'iterations': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['dataset', 'server', 'time_mean', 'time_std', 'iter_mean', 'iter_std']
        grouped['time_std'] = grouped['time_std'].fillna(0)
        grouped['iter_std'] = grouped['iter_std'].fillna(0)
        
        # Get unique datasets
        datasets = sorted(grouped['dataset'].unique())
        if len(datasets) == 0:
            continue
        
        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a2e')
        
        x = np.arange(len(datasets))
        width = 0.35
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
        
        # Solver Time chart
        cpp_times = grouped[grouped['server'] == 'cpp'].set_index('dataset').reindex(datasets)
        py_times = grouped[grouped['server'] == 'python'].set_index('dataset').reindex(datasets)
        
        if not cpp_times.empty:
            ax1.bar(x - width/2, cpp_times['time_mean'].fillna(0), width, 
                   yerr=cpp_times['time_std'].fillna(0), label='C++', color='#f97316', capsize=5)
        if not py_times.empty:
            ax1.bar(x + width/2, py_times['time_mean'].fillna(0), width,
                   yerr=py_times['time_std'].fillna(0), label='Python', color='#3b82f6', capsize=5)
        
        ax1.set_xlabel('Dataset', color='white', fontsize=12)
        ax1.set_ylabel('Tempo (ms)', color='white', fontsize=12)
        ax1.set_title(f'Tempo de Solver - {size_label}', color='white', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.split('_')[-1] for d in datasets], color='white')
        ax1.legend(facecolor='#252542', edgecolor='white', labelcolor='white')
        
        # Iterations chart
        if not cpp_times.empty:
            ax2.bar(x - width/2, cpp_times['iter_mean'].fillna(0), width,
                   yerr=cpp_times['iter_std'].fillna(0), label='C++', color='#f97316', capsize=5)
        if not py_times.empty:
            ax2.bar(x + width/2, py_times['iter_mean'].fillna(0), width,
                   yerr=py_times['iter_std'].fillna(0), label='Python', color='#3b82f6', capsize=5)
        
        ax2.set_xlabel('Dataset', color='white', fontsize=12)
        ax2.set_ylabel('Iterações', color='white', fontsize=12)
        ax2.set_title(f'Iterações até Convergência - {size_label}', color='white', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.split('_')[-1] for d in datasets], color='white')
        ax2.legend(facecolor='#252542', edgecolor='white', labelcolor='white')
        
        plt.tight_layout()
        filename = f"race_comparison_{size_label}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        result[size_label] = filename
    
    return result


def generate_html_report(input_dir: Path, output_dir: Path):
    """Generate beautiful HTML report with CSS styling."""
    
    config = get_config_values(input_dir)
    epsilon = config.get('settings', {}).get('epsilon_tolerance', 1e-4)
    
    # Load job metrics
    metrics_csv = input_dir / 'telemetry' / 'job_metrics.csv'
    if not metrics_csv.exists():
        print(f"[ERROR] Metrics file not found: {metrics_csv}")
        return None
    
    df = pd.read_csv(metrics_csv)
    
    # Load environment
    env_file = input_dir / 'data' / 'environment.json'
    env = {}
    if env_file.exists():
        with open(env_file) as f:
            env = json.load(f)
    
    # Collect all unique images
    img_dir = input_dir / 'images'
    all_images = []
    
    if img_dir.exists():
        for meta_path in sorted(img_dir.glob('*_meta.json')):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                csv_path = meta_path.parent / meta_path.name.replace('_meta.json', '_image.csv')
                if csv_path.exists():
                    all_images.append({
                        'csv': csv_path,
                        'meta': meta,
                        'key': (meta.get('dataset_id', ''), meta.get('server', ''))
                    })
            except:
                pass
    
    # Deduplicate: keep best error per (dataset, server)
    unique_images = {}
    for img in all_images:
        key = img['key']
        if key not in unique_images:
            unique_images[key] = img
        else:
            if img['meta'].get('final_error', float('inf')) < unique_images[key]['meta'].get('final_error', float('inf')):
                unique_images[key] = img
    
    sorted_keys = sorted(unique_images.keys())
    images = [unique_images[k] for k in sorted_keys]
    
    # Create assets directory
    assets_dir = output_dir / 'report_assets'
    assets_dir.mkdir(exist_ok=True)
    
    # Generate images and GIFs
    print(f"[INFO] Generating {len(images)} images...")
    image_data = []
    
    for i, img in enumerate(images):
        meta = img['meta']
        dataset = meta.get('dataset_id', 'unknown')
        server = meta.get('server', 'unknown').lower()
        error = meta.get('final_error', 0)
        
        png_name = f"img_{i:02d}_{dataset}_{server}.png"
        png_path = assets_dir / png_name
        
        create_image_png(img['csv'], png_path, dataset, error, epsilon)
        
        # Create iteration frames for carousel using job_id
        job_id = meta.get('job_id', '')
        iter_frames = create_iteration_frames(img_dir, assets_dir, job_id, server)
        
        image_data.append({
            'png': png_name,
            'frames': iter_frames,  # List of frame filenames for carousel
            'dataset': dataset,
            'server': server.upper(),
            'error': error,
            'error_fmt': format_error(error),
            'iterations': meta.get('iterations', 0),
            'solver_time': meta.get('solver_time_ms', 0),
            'converged': meta.get('converged', False),
            'is_good': error < epsilon
        })
    
    # Generate L-curve plots for each unique (dataset, server)
    print("[INFO] Generating L-curve plots...")
    lcurve_data = []
    for img in images:
        meta = img['meta']
        job_id = meta.get('job_id', '')
        dataset = meta.get('dataset_id', 'unknown')
        server = meta.get('server', 'unknown').lower()
        
        lcurve_file = create_lcurve_plot(img_dir, assets_dir, job_id, server, dataset)
        if lcurve_file:
            lcurve_data.append({
                'file': lcurve_file,
                'dataset': dataset,
                'server': server.upper()
            })
    
    # Generate resource usage charts
    print("[INFO] Generating resource usage charts...")
    resource_charts = create_resource_usage_charts(input_dir, assets_dir)
    
    # Generate race comparison charts
    print("[INFO] Generating race comparison charts...")
    race_charts = create_race_comparison_charts(df, assets_dir)
    
    # Generate per-job memory chart
    print("[INFO] Generating per-job memory chart...")
    per_job_memory = create_per_job_memory_chart(df, assets_dir)
    
    # Calculate speedup
    cpp_times = df[df['server'] == 'cpp']['solver_time_ms']
    py_times = df[df['server'] == 'python']['solver_time_ms']
    speedup = py_times.mean() / cpp_times.mean() if len(cpp_times) > 0 and len(py_times) > 0 else 1.0
    
    # Generate performance table rows
    perf_rows = []
    for (ds, srv), group in df.groupby(['dataset_id', 'server']):
        avg_time = group['solver_time_ms'].mean()
        avg_err = group['final_error'].mean()
        avg_iter = group['iterations'].mean()
        perf_rows.append({
            'dataset': ds,
            'server': srv.upper(),
            'time': f"{avg_time:.1f}",
            'error': format_error(avg_err),
            'iterations': int(avg_iter)
        })
    
    # Generate HTML
    current_date = datetime.now().strftime('%d/%m/%Y %H:%M')
    html = generate_html_template(
        title=f"Relatório execução: {current_date} | {input_dir.name}",
        date=current_date,
        dir_name=input_dir.name,
        env=env,
        epsilon=epsilon,
        total_jobs=len(df),
        datasets=', '.join(df['dataset_id'].unique()),
        servers=', '.join(df['server'].unique()),
        perf_rows=perf_rows,
        images=image_data,
        speedup=speedup,
        lcurve_data=lcurve_data,
        resource_charts=resource_charts,
        race_charts=race_charts,
        per_job_memory=per_job_memory
    )
    
    html_path = output_dir / 'Relatorio_Cientifico.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"[SUCCESS] HTML report: {html_path}")
    return html_path


def generate_html_template(**data):
    """Generate the HTML template with modern CSS."""
    
    # Build performance table rows
    perf_table = ""
    for row in data['perf_rows']:
        perf_table += f"""
            <tr>
                <td>{row['dataset']}</td>
                <td><span class="badge {'badge-cpp' if row['server']=='CPP' else 'badge-py'}">{row['server']}</span></td>
                <td>{row['time']}</td>
                <td class="mono">{row['error']}</td>
                <td>{row['iterations']}</td>
            </tr>"""
    
    # Build L-curve section
    lcurve_section = ""
    if data.get('lcurve_data'):
        lcurve_cards = ""
        for lc in data['lcurve_data']:
            lcurve_cards += f'''
            <div class="chart-card">
                <h4>{lc['dataset']} - <span class="badge {'badge-cpp' if lc['server']=='CPP' else 'badge-py'}">{lc['server']}</span></h4>
                <img src="report_assets/{lc['file']}" alt="L-curve {lc['dataset']}" class="chart-img">
            </div>'''
        lcurve_section = f'''
        <section>
            <h2>📈 Curvas L (Análise de Regularização)</h2>
            <p>O "cotovelo" da curva L indica o ponto ótimo de regularização - balanço entre erro e norma da solução.</p>
            <div class="chart-grid">
                {lcurve_cards}
            </div>
        </section>'''
    
    # Build resource usage section
    resource_section = ""
    if data.get('resource_charts') or data.get('per_job_memory'):
        rc = data.get('resource_charts', {})
        cpu_html = f'<img src="report_assets/{rc["cpu"]}" alt="CPU Usage" class="chart-wide">' if 'cpu' in rc else ''
        mem_html = f'<img src="report_assets/{rc["memory"]}" alt="Memory Usage" class="chart-wide">' if 'memory' in rc else ''
        job_mem_html = f'<img src="report_assets/{data["per_job_memory"]}" alt="Per-Job Memory" class="chart-wide">' if data.get('per_job_memory') else ''
        
        resource_section = f'''
        <section>
            <h2>⚡ Uso de Recursos</h2>
            <p>Monitoramento de recursos durante a execução.</p>
            <div class="resource-charts">
                {job_mem_html}
                {cpu_html}
                {mem_html}
            </div>
        </section>'''
    
    # Build race comparison section
    race_section = ""
    if data.get('race_charts'):
        race_cards = ""
        for size, filename in data['race_charts'].items():
            race_cards += f'''
            <div class="race-chart">
                <h4>Comparação {size}</h4>
                <img src="report_assets/{filename}" alt="Race {size}" class="chart-wide">
            </div>'''
        race_section = f'''
        <section>
            <h2>🏁 Comparativo Race (C++ vs Python)</h2>
            <p>Barras de erro indicam desvio padrão calculado a partir de múltiplas execuções.</p>
            {race_cards}
        </section>'''
    
    # Build image cards
    image_cards = ""
    for i, img in enumerate(data['images']):
        status_class = "status-good" if img['is_good'] else "status-bad"
        status_icon = "✓" if img['is_good'] else "✗"
        status_text = "Convergiu" if img['converged'] else "Max Iter"
        
        # Build carousel if frames exist
        carousel_html = ""
        if img['frames'] and len(img['frames']) > 0:
            num_frames = len(img['frames'])
            carousel_html = f'''
                <div class="carousel-controls">
                    <input type="range" class="carousel-slider" id="slider-{i}" 
                           min="0" max="{num_frames}" value="{num_frames}" 
                           oninput="slideFrame({i}, this.value)">
                    <span class="carousel-info" id="carousel-info-{i}">Final</span>
                </div>'''
        
        image_cards += f'''
        <div class="image-card" data-frames='{json.dumps(img["frames"])}' data-main="{img['png']}" id="card-{i}">
            <div class="card-header">
                <h3>{img['dataset']}</h3>
                <span class="badge {'badge-cpp' if img['server']=='CPP' else 'badge-py'}">{img['server']}</span>
            </div>
            <div class="image-container">
                <img src="report_assets/{img['png']}" alt="{img['dataset']}" class="main-image" id="img-{i}">
                {carousel_html}
            </div>
            <div class="card-stats">
                <div class="stat">
                    <span class="stat-label">Erro Final</span>
                    <span class="stat-value {status_class}">{status_icon} {img['error_fmt']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Tempo</span>
                    <span class="stat-value">{img['solver_time']:.1f} ms</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Iterações</span>
                    <span class="stat-value">{img['iterations']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Status</span>
                    <span class="stat-value">{status_text}</span>
                </div>
            </div>
        </div>'''
    
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['title']}</title>
    <style>
        :root {{
            --bg-dark: #0f0f1a;
            --bg-card: #1a1a2e;
            --bg-hover: #252542;
            --accent: #7c3aed;
            --accent-light: #a78bfa;
            --text: #e2e8f0;
            --text-dim: #94a3b8;
            --success: #22c55e;
            --error: #ef4444;
            --cpp: #f97316;
            --python: #3b82f6;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--bg-card), var(--accent) 150%);
            padding: 3rem 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--text), var(--accent-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{ color: var(--text-dim); font-size: 1.1rem; }}
        
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        
        .meta-item {{
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 0.5rem;
        }}
        
        .meta-label {{ font-size: 0.8rem; color: var(--text-dim); }}
        .meta-value {{ font-size: 1.2rem; font-weight: 600; }}
        
        section {{
            background: var(--bg-card);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        h2 {{
            color: var(--accent-light);
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-hover);
        }}
        
        th {{
            background: var(--bg-hover);
            color: var(--accent-light);
            font-weight: 600;
        }}
        
        tr:hover {{ background: var(--bg-hover); }}
        
        .mono {{ font-family: 'Consolas', monospace; }}
        
        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .badge-cpp {{ background: var(--cpp); color: white; }}
        .badge-py {{ background: var(--python); color: white; }}
        
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
        }}
        
        .image-card {{
            background: var(--bg-hover);
            border-radius: 1rem;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(124, 58, 237, 0.2);
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: var(--bg-card);
        }}
        
        .card-header h3 {{ font-size: 1.2rem; }}
        
        .image-container {{
            position: relative;
            background: #000;
            aspect-ratio: 1 / 1;
            overflow: hidden;
        }}
        
        .main-image {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }}
        
        .gif-toggle {{
            position: absolute;
            bottom: 1rem;
            right: 1rem;
        }}
        
        .btn-anim {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.3s;
        }}
        
        .btn-anim:hover {{ background: var(--accent-light); }}
        
        .card-stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            padding: 1rem 1.5rem;
        }}
        
        .stat {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{ font-size: 0.75rem; color: var(--text-dim); }}
        .stat-value {{ font-weight: 600; }}
        
        .status-good {{ color: var(--success); }}
        .status-bad {{ color: var(--error); }}
        
        .speedup {{
            font-size: 3rem;
            font-weight: 700;
            color: var(--accent-light);
            text-align: center;
            padding: 2rem;
        }}
        
        .tech-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        
        .tech-card {{
            background: var(--bg-hover);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 4px solid var(--accent);
        }}
        
        .tech-card h4 {{ color: var(--accent-light); margin-bottom: 0.5rem; }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-dim);
            font-size: 0.9rem;
        }}
        
        .legend {{
            display: flex;
            gap: 2rem;
            margin: 1rem 0;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .dot-good {{ background: var(--success); }}
        .dot-bad {{ background: var(--error); }}
        
        /* Carousel slider styles */
        .carousel-controls {{
            position: absolute;
            bottom: 0.5rem;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.3rem;
            background: rgba(0,0,0,0.75);
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            min-width: 200px;
        }}
        
        .carousel-slider {{
            width: 100%;
            height: 6px;
            background: #333;
            border-radius: 3px;
            cursor: pointer;
            -webkit-appearance: none;
            appearance: none;
        }}
        
        .carousel-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        
        .carousel-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        
        .carousel-slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: var(--accent);
            border-radius: 50%;
            border: none;
            cursor: pointer;
        }}
        
        .carousel-info {{
            color: white;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        /* Chart sections */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        
        .chart-card {{
            background: var(--bg-hover);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
        }}
        
        .chart-card h4 {{
            color: var(--text);
            margin-bottom: 0.5rem;
        }}
        
        .chart-img {{
            width: 100%;
            max-width: 500px;
            border-radius: 0.5rem;
        }}
        
        .chart-wide {{
            width: 100%;
            max-width: 900px;
            border-radius: 0.5rem;
            margin: 1rem auto;
            display: block;
        }}
        
        .resource-charts {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
        }}
        
        .race-chart {{
            margin: 1.5rem 0;
            text-align: center;
        }}
        
        .race-chart h4 {{
            color: var(--accent-light);
            margin-bottom: 0.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 {data['title']}</h1>
            
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">Plataforma</div>
                    <div class="meta-value">{data['env'].get('platform', 'N/A')[:30]}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">CPU</div>
                    <div class="meta-value">{data['env'].get('cpu_cores_logical', 'N/A')} cores</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">RAM</div>
                    <div class="meta-value">{data['env'].get('ram_total_gb', 'N/A')} GB</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Total Jobs</div>
                    <div class="meta-value">{data['total_jobs']}</div>
                </div>
            </div>
        </header>
        
        <section>
            <h2>📊 Comparativo de Desempenho</h2>
            <div class="speedup">Speedup C++ vs Python: {data['speedup']:.2f}x</div>
            <table>
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>Servidor</th>
                        <th title="Tempo médio de execução do solver em milissegundos. Menor = melhor desempenho.">Tempo Médio (ms) ⓘ</th>
                        <th title="Erro residual final ||Hx - g||. Menor = solução mais precisa. Calculado como norma do resíduo.">Erro Médio ⓘ</th>
                        <th title="Número de iterações até convergência. Menos iterações com baixo erro = melhor eficiência.">Iterações ⓘ</th>
                    </tr>
                </thead>
                <tbody>
                    {perf_table}
                </tbody>
            </table>
        </section>
        
        
        <section>
            <h2>🖼️ Galeria de Imagens Reconstruídas</h2>
            <p>Total: {len(data['images'])} imagens | Threshold: {data['epsilon']:.0e}</p>
            <div class="legend">
                <div class="legend-item"><span class="dot dot-good"></span> Erro &lt; {data['epsilon']:.0e}</div>
                <div class="legend-item"><span class="dot dot-bad"></span> Erro ≥ {data['epsilon']:.0e}</div>
            </div>
            <div class="image-grid">
                {image_cards}
            </div>
        </section>
        
        {lcurve_section}
        
        {race_section}
        
        {resource_section}
        
        <footer>
            Relatório gerado automaticamente em {datetime.now().isoformat()}
        </footer>
    </div>
    
    <script>
        // Carousel state: track current index for each card
        const carouselState = {{}};
        
        function getCardData(cardIndex) {{
            const card = document.getElementById('card-' + cardIndex);
            if (!card) return null;
            const frames = JSON.parse(card.dataset.frames || '[]');
            const mainImg = card.dataset.main;
            return {{ frames, mainImg, card }};
        }}
        
        function slideFrame(cardIndex, value) {{
            const data = getCardData(cardIndex);
            if (!data) return;
            
            const img = document.getElementById('img-' + cardIndex);
            const info = document.getElementById('carousel-info-' + cardIndex);
            const idx = parseInt(value);
            
            if (idx >= data.frames.length) {{
                // Show final image
                img.src = 'report_assets/' + data.mainImg;
                info.textContent = 'Final';
            }} else {{
                // Show iteration frame
                img.src = 'report_assets/' + data.frames[idx];
                info.textContent = 'Iteração ' + (idx + 1) + '/' + data.frames.length;
            }}
        }}
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='Generate HTML Scientific Report')
    parser.add_argument('--input-dir', type=str, required=True, help='Execution directory')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: same as input)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    print(f"[INFO] Generating HTML report for: {input_dir.name}")
    generate_html_report(input_dir, output_dir)


if __name__ == '__main__':
    main()
