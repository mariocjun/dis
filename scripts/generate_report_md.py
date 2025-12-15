#!/usr/bin/env python3
"""
generate_report_md.py - Markdown-based Scientific Report Generator
Generates a clean Markdown report and converts to PDF using available tools.
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


def get_error_threshold(input_dir: Path) -> float:
    """Get epsilon_tolerance from config file as error threshold."""
    config_path = input_dir / 'data' / 'config_snapshot.yaml'
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('settings', {}).get('epsilon_tolerance', 1e-4)
        except:
            pass
    return 1e-4  # Default


def create_image_png(csv_path: Path, output_path: Path, title: str, error: float, threshold: float):
    """Create PNG from CSV image data."""
    data = np.loadtxt(csv_path, delimiter=',')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    v_min, v_max = np.min(data), np.max(data)
    if v_max <= v_min:
        v_max = v_min + 1.0
    
    im = ax.imshow(data, cmap='inferno', vmin=v_min, vmax=v_max,
                   aspect='equal', interpolation='nearest')
    
    # Color based on error vs threshold
    title_color = "green" if error < threshold else "red"
    ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
    
    fig.colorbar(im, ax=ax, label='Intensidade')
    ax.axis('off')
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def generate_markdown_report(input_dir: Path, output_dir: Path):
    """Generate Markdown report with all images and statistics."""
    
    # Get error threshold from config
    ERROR_THRESHOLD = get_error_threshold(input_dir)
    print(f"[INFO] Using epsilon_tolerance as threshold: {ERROR_THRESHOLD}")
    
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
    
    # Sort by name
    sorted_keys = sorted(unique_images.keys())
    images = [unique_images[k] for k in sorted_keys]
    
    # Create image PNGs
    report_images_dir = output_dir / 'report_images'
    report_images_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i, img in enumerate(images):
        meta = img['meta']
        dataset = meta.get('dataset_id', 'unknown')
        server = meta.get('server', 'unknown').upper()
        error = meta.get('final_error', 0)
        
        png_name = f"img_{i:02d}_{dataset}_{server.lower()}.png"
        png_path = report_images_dir / png_name
        
        title = f"{dataset} [{server}] - Erro: {error:.4e}"
        create_image_png(img['csv'], png_path, title, error, ERROR_THRESHOLD)
        
        image_paths.append({
            'path': png_path,
            'dataset': dataset,
            'server': server,
            'error': error,
            'iterations': meta.get('iterations', 0),
            'solver_time': meta.get('solver_time_ms', 0),
            'converged': meta.get('converged', False)
        })
    
    # Generate Markdown
    md_lines = []
    
    # Header
    md_lines.append("# Relatório Científico: Reconstrução de Imagens Ultrassônicas")
    md_lines.append("")
    md_lines.append(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    md_lines.append(f"**Diretório:** `{input_dir.name}`")
    md_lines.append("")
    
    # Environment
    md_lines.append("## Ambiente de Execução")
    md_lines.append("")
    md_lines.append(f"- **Plataforma:** {env.get('platform', 'N/A')}")
    md_lines.append(f"- **CPU:** {env.get('cpu_cores_logical', 'N/A')} cores lógicos")
    md_lines.append(f"- **RAM:** {env.get('ram_total_gb', 'N/A')} GB")
    md_lines.append(f"- **Python:** {env.get('python_version', 'N/A')}")
    md_lines.append("")
    
    # Statistics Summary
    md_lines.append("## Resumo Estatístico")
    md_lines.append("")
    md_lines.append(f"- **Total de Jobs:** {len(df)}")
    md_lines.append(f"- **Datasets:** {', '.join(df['dataset_id'].unique())}")
    md_lines.append(f"- **Servidores:** {', '.join(df['server'].unique())}")
    md_lines.append("")
    
    # Performance Table
    md_lines.append("### Comparativo de Desempenho")
    md_lines.append("")
    md_lines.append("| Dataset | Servidor | Tempo Médio (ms) | Erro Médio | Iterações |")
    md_lines.append("|---------|----------|------------------|------------|-----------|")
    
    for (ds, srv), group in df.groupby(['dataset_id', 'server']):
        avg_time = group['solver_time_ms'].mean()
        avg_err = group['final_error'].mean()
        avg_iter = group['iterations'].mean()
        md_lines.append(f"| {ds} | {srv.upper()} | {avg_time:.1f} | {avg_err:.4f} | {avg_iter:.0f} |")
    
    md_lines.append("")
    
    # GIL/JIT/OpenMP explanation
    md_lines.append("## Análise Técnica: Python vs C++")
    md_lines.append("")
    md_lines.append("### Global Interpreter Lock (GIL)")
    md_lines.append("O GIL é um mutex que impede múltiplas threads Python de executarem bytecode simultaneamente.")
    md_lines.append("**Impacto:** Python NÃO usa múltiplos cores mesmo com threading.")
    md_lines.append("")
    md_lines.append("### OpenMP (C++)")
    md_lines.append("OpenMP permite paralelismo real via `#pragma`:")
    md_lines.append("- **H·p** (produto matriz-vetor) → threads paralelas")
    md_lines.append("- **Normas** → redução paralela")
    md_lines.append("")
    md_lines.append("### Tabela Comparativa")
    md_lines.append("")
    md_lines.append("| Aspecto | Python | C++ |")
    md_lines.append("|---------|--------|-----|")
    md_lines.append("| Execução | Interpretada | Compilada |")
    md_lines.append("| Threads | GIL bloqueia | OpenMP real |")
    md_lines.append("| I/O | Parse CSV | Load binário |")
    md_lines.append("")
    
    # Images Section
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Galeria de Imagens Reconstruídas")
    md_lines.append("")
    md_lines.append(f"**Total:** {len(image_paths)} imagens únicas (melhor por dataset+servidor)")
    md_lines.append("")
    md_lines.append("**Legenda de cores:**")
    md_lines.append(f"- 🟢 **Verde:** Erro < {ERROR_THRESHOLD} (aceitável)")
    md_lines.append(f"- 🔴 **Vermelho:** Erro ≥ {ERROR_THRESHOLD} (divergência)")
    md_lines.append("")
    
    for i, img_info in enumerate(image_paths):
        error_color = "🟢" if img_info['error'] < ERROR_THRESHOLD else "🔴"
        status = "Convergiu" if img_info['converged'] else "Max Iterações"
        
        md_lines.append(f"### {i+1}. {img_info['dataset']} [{img_info['server']}]")
        md_lines.append("")
        md_lines.append(f"![{img_info['dataset']}]({img_info['path'].name})")
        md_lines.append("")
        md_lines.append(f"- **Erro Final:** {error_color} `{img_info['error']:.4e}`")
        md_lines.append(f"- **Tempo:** {img_info['solver_time']:.1f} ms")
        md_lines.append(f"- **Iterações:** {img_info['iterations']}")
        md_lines.append(f"- **Status:** {status}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Conclusion
    md_lines.append("## Conclusões")
    md_lines.append("")
    
    # Calculate speedup
    cpp_times = df[df['server'] == 'cpp']['solver_time_ms']
    py_times = df[df['server'] == 'python']['solver_time_ms']
    
    if len(cpp_times) > 0 and len(py_times) > 0:
        speedup = py_times.mean() / cpp_times.mean()
        md_lines.append(f"1. **Speedup C++ vs Python:** {speedup:.2f}x")
    
    md_lines.append(f"2. **Total de Reconstruções:** {len(df)} jobs processados")
    md_lines.append(f"3. **Imagens Únicas:** {len(image_paths)} combinações dataset+servidor")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append(f"*Relatório gerado automaticamente em {datetime.now().isoformat()}*")
    
    # Save Markdown
    md_path = output_dir / 'Relatorio_Cientifico.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"[SUCCESS] Markdown report: {md_path}")
    
    # Try to convert to PDF
    pdf_path = convert_md_to_pdf(md_path, report_images_dir)
    
    return md_path, pdf_path


def convert_md_to_pdf(md_path: Path, images_dir: Path):
    """Try multiple methods to convert Markdown to PDF."""
    import subprocess
    import shutil
    
    pdf_path = md_path.with_suffix('.pdf')
    
    # Method 1: pandoc (best quality)
    if shutil.which('pandoc'):
        try:
            result = subprocess.run([
                'pandoc', str(md_path),
                '-o', str(pdf_path),
                '--resource-path', str(images_dir),
                '-V', 'geometry:margin=1in'
            ], capture_output=True, text=True, cwd=str(images_dir))
            if result.returncode == 0:
                print(f"[SUCCESS] PDF (pandoc): {pdf_path}")
                return pdf_path
        except Exception as e:
            print(f"[WARN] Pandoc failed: {e}")
    
    # Method 2: Use markdown + matplotlib to create simple PDF
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with PdfPages(pdf_path) as pdf:
            # Simple text rendering
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.axis('off')
            
            # Just render the Markdown as text on first page
            text = ''.join(lines[:50])  # First 50 lines
            ax.text(0, 1, text, fontsize=8, verticalalignment='top', fontfamily='monospace',
                   transform=ax.transAxes, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Add image pages
            for img_file in sorted(images_dir.glob('*.png')):
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(img_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(img_file.stem, fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"[SUCCESS] PDF (fallback): {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"[WARN] PDF creation failed: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate Markdown Scientific Report')
    parser.add_argument('--input-dir', type=str, required=True, help='Execution directory')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: same as input)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    print(f"[INFO] Generating Markdown report for: {input_dir.name}")
    generate_markdown_report(input_dir, output_dir)


if __name__ == '__main__':
    main()
