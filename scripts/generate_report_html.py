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


def create_iteration_gif(img_dir: Path, output_dir: Path, dataset_id: str, server: str) -> Path:
    """Create GIF animation from iteration images if available."""
    # Look for iteration files pattern: *_iter_*.csv
    pattern = f"*{dataset_id}*{server}*_iter_*.csv"
    iter_files = sorted(glob.glob(str(img_dir / pattern)))
    
    if len(iter_files) < 2:
        return None
    
    gif_path = output_dir / f"anim_{dataset_id}_{server}.gif"
    frames = []
    
    for iter_file in iter_files:
        try:
            data = np.loadtxt(iter_file, delimiter=',')
            
            fig, ax = plt.subplots(figsize=(5, 5))
            v_min, v_max = np.min(data), np.max(data)
            if v_max <= v_min:
                v_max = v_min + 1.0
            
            # Rotate 90 degrees counter-clockwise (left)
            data_rotated = np.rot90(data, k=1)
            
            ax.imshow(data_rotated, cmap='inferno', vmin=v_min, vmax=v_max)
            iter_num = Path(iter_file).stem.split('_')[-1]
            ax.set_title(f"Iteração {iter_num}", fontsize=12)
            ax.axis('off')
            
            # Save to buffer
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)
        except:
            pass
    
    if frames:
        imageio.mimsave(gif_path, frames, fps=3, loop=0)
        return gif_path
    return None


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
        
        # Try to create GIF animation
        gif_path = create_iteration_gif(img_dir, assets_dir, dataset, server)
        
        image_data.append({
            'png': png_name,
            'gif': gif_path.name if gif_path else None,
            'dataset': dataset,
            'server': server.upper(),
            'error': error,
            'error_fmt': format_error(error),
            'iterations': meta.get('iterations', 0),
            'solver_time': meta.get('solver_time_ms', 0),
            'converged': meta.get('converged', False),
            'is_good': error < epsilon
        })
    
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
    html = generate_html_template(
        title="Relatório Científico: Reconstrução Ultrassônica",
        date=datetime.now().strftime('%d/%m/%Y %H:%M'),
        dir_name=input_dir.name,
        env=env,
        epsilon=epsilon,
        total_jobs=len(df),
        datasets=', '.join(df['dataset_id'].unique()),
        servers=', '.join(df['server'].unique()),
        perf_rows=perf_rows,
        images=image_data,
        speedup=speedup
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
    
    # Build image cards
    image_cards = ""
    for i, img in enumerate(data['images']):
        status_class = "status-good" if img['is_good'] else "status-bad"
        status_icon = "✓" if img['is_good'] else "✗"
        status_text = "Convergiu" if img['converged'] else "Max Iter"
        
        gif_html = ""
        if img['gif']:
            gif_html = f"""
                <div class="gif-toggle">
                    <button onclick="toggleGif(this, '{img['gif']}')" class="btn-anim">▶ Animação</button>
                </div>"""
        
        image_cards += f"""
        <div class="image-card">
            <div class="card-header">
                <h3>{img['dataset']}</h3>
                <span class="badge {'badge-cpp' if img['server']=='CPP' else 'badge-py'}">{img['server']}</span>
            </div>
            <div class="image-container">
                <img src="report_assets/{img['png']}" alt="{img['dataset']}" class="main-image" id="img-{i}">
                {gif_html}
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
        </div>"""
    
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
        }}
        
        .main-image {{
            width: 100%;
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 {data['title']}</h1>
            <p class="subtitle">{data['date']} | {data['dir_name']}</p>
            
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
                        <th>Tempo Médio (ms)</th>
                        <th>Erro Médio</th>
                        <th>Iterações</th>
                    </tr>
                </thead>
                <tbody>
                    {perf_table}
                </tbody>
            </table>
        </section>
        
        <section>
            <h2>⚙️ Análise Técnica</h2>
            <div class="tech-grid">
                <div class="tech-card">
                    <h4>🐍 Python (GIL)</h4>
                    <p>O Global Interpreter Lock impede paralelismo real. Threads Python compartilham um único lock.</p>
                </div>
                <div class="tech-card">
                    <h4>⚡ C++ (OpenMP)</h4>
                    <p>Paralelismo nativo via #pragma. Operações H·p distribuídas entre threads reais.</p>
                </div>
                <div class="tech-card">
                    <h4>📈 Eigen</h4>
                    <p>Biblioteca C++ com vetorização SIMD automática para operações matriciais.</p>
                </div>
                <div class="tech-card">
                    <h4>💾 Cache Binário</h4>
                    <p>C++ usa arquivos .bin pré-processados. Zero parsing CSV em runtime.</p>
                </div>
            </div>
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
        
        <footer>
            Relatório gerado automaticamente em {datetime.now().isoformat()}
        </footer>
    </div>
    
    <script>
        function toggleGif(btn, gifName) {{
            const container = btn.closest('.image-container');
            const img = container.querySelector('.main-image');
            const originalSrc = img.dataset.original || img.src;
            
            if (img.src.includes('.gif')) {{
                img.src = originalSrc;
                btn.textContent = '▶ Animação';
            }} else {{
                img.dataset.original = img.src;
                img.src = 'report_assets/' + gifName;
                btn.textContent = '⏹ Parar';
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
