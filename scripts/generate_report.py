#!/usr/bin/env python3
"""
generate_report.py - Scientific Report Generator (Academic Quality PDF)
=======================================================================
Generates comprehensive PDF reports following Maziero's OS scheduling
visualization style with proper Gantt charts, queue visualization,
and scientific article-level documentation.

Version: 2.1 - Fixed pagination, balanced image selection, added explanations
"""

import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import platform
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import seaborn as sns
    HAS_LIB = True
except ImportError as e:
    HAS_LIB = False
    print(f"[FATAL] Missing libraries: {e}")
    print("Run: pip install pandas matplotlib seaborn numpy")
    sys.exit(1)

# =============================================================================
# CONFIGURAÇÃO ESTÉTICA - Estilo Artigo Científico
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Color palette
COLORS = {
    'python': '#3498DB',      # Azul
    'cpp': '#E74C3C',         # Vermelho
    'queue': '#95A5A6',       # Cinza (fila)
    'arrival': '#2ECC71',     # Verde (chegada)
    'background': '#FAFAFA',  # Fundo
    'grid': '#E0E0E0',        # Grade
    'text': '#2C3E50',        # Texto
    'header': '#1A5276',      # Cabeçalho
}


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
def load_data(input_dir: Path):
    """Load telemetry data from the experiment directory."""
    metrics_file = input_dir / 'telemetry' / 'job_metrics.csv'
    sys_file = input_dir / 'telemetry' / 'system_metrics.csv'
    env_file = input_dir / 'data' / 'environment.json'
    
    if not metrics_file.exists():
        print(f"[ERROR] Metrics file not found: {metrics_file}")
        return None, None, None
    
    df_jobs = pd.read_csv(metrics_file)
    df_sys = pd.read_csv(sys_file) if sys_file.exists() else pd.DataFrame()
    
    env = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            env = json.load(f)
    
    return df_jobs, df_sys, env


def preprocess_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Process job data for visualization."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Parse timestamps
    df['end_dt'] = pd.to_datetime(df['timestamp_end'])
    df['start_dt'] = pd.to_datetime(df['timestamp_start'])
    
    # Calculate derived metrics
    df['solver_s'] = df['solver_time_ms'] / 1000.0
    df['latency_s'] = df['latency_ms'] / 1000.0
    
    # Compute execution start (end - solver_time)
    df['exec_start'] = df['end_dt'] - pd.to_timedelta(df['solver_time_ms'], unit='ms')
    
    # Normalize to T=0
    t0 = df['start_dt'].min()
    df['t_arrival'] = (df['start_dt'] - t0).dt.total_seconds()
    df['t_exec_start'] = (df['exec_start'] - t0).dt.total_seconds()
    df['t_end'] = (df['end_dt'] - t0).dt.total_seconds()
    
    # Queue duration
    df['queue_duration'] = df['t_exec_start'] - df['t_arrival']
    df['exec_duration'] = df['t_end'] - df['t_exec_start']
    
    return df.sort_values('t_arrival')


# =============================================================================
# PÁGINA 1: CAPA E RESUMO EXECUTIVO
# =============================================================================
def create_cover_page(df: pd.DataFrame, env: dict, pdf: PdfPages, input_dir: Path):
    """Create a professional cover page with executive summary."""
    fig = plt.figure(figsize=(8.5, 11))
    
    # Header bar
    ax_header = fig.add_axes([0, 0.85, 1, 0.15])
    ax_header.set_facecolor(COLORS['header'])
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis('off')
    
    # Title
    ax_header.text(0.5, 0.6, 'RELATÓRIO CIENTÍFICO', ha='center', va='center',
                   fontsize=24, fontweight='bold', color='white', fontfamily='serif')
    ax_header.text(0.5, 0.25, 'Benchmark de Reconstrução Tomográfica por Ultrassom',
                   ha='center', va='center', fontsize=14, color='white', fontfamily='serif')
    
    # Main content area
    ax_main = fig.add_axes([0.1, 0.1, 0.8, 0.72])
    ax_main.axis('off')
    
    # Date and Run ID
    run_name = input_dir.name if input_dir.name else 'N/A'
    date_str = datetime.now().strftime('%d de %B de %Y às %H:%M')
    
    content = []
    content.append(f"Data de Geração: {date_str}")
    content.append(f"Identificador da Execução: {run_name}")
    content.append("")
    content.append("─" * 50)
    content.append("RESUMO EXECUTIVO")
    content.append("─" * 50)
    content.append("")
    
    # Calculate statistics
    if not df.empty:
        py_data = df[df['server'] == 'python']['solver_time_ms']
        cpp_data = df[df['server'] == 'cpp']['solver_time_ms']
        
        py_mean = py_data.mean() if len(py_data) > 0 else 0
        cpp_mean = cpp_data.mean() if len(cpp_data) > 0 else 0
        speedup = py_mean / cpp_mean if cpp_mean > 0 else 0
        
        total_jobs = len(df)
        py_jobs = len(py_data)
        cpp_jobs = len(cpp_data)
        
        content.append(f"Total de Jobs: {total_jobs} (Python: {py_jobs}, C++: {cpp_jobs})")
        content.append("")
        content.append(f"Tempo Médio de Solver:")
        content.append(f"  Python: {py_mean:.2f} ms (σ={py_data.std():.2f})")
        content.append(f"  C++:    {cpp_mean:.2f} ms (σ={cpp_data.std():.2f})")
        content.append("")
        content.append(f"SPEEDUP GLOBAL: {speedup:.2f}x")
        content.append("")
        
        converged = df['converged'].astype(str).str.lower().isin(['true', '1', 'yes'])
        conv_rate = converged.sum() / len(df) * 100
        content.append(f"Taxa de Convergência: {conv_rate:.1f}%")
        content.append(f"Erro Final Médio: {df['final_error'].mean():.4e}")
    
    content.append("")
    content.append("─" * 50)
    content.append("AMBIENTE DE TESTE")
    content.append("─" * 50)
    content.append("")
    
    if env:
        content.append(f"CPU: {env.get('processor', 'N/A')}")
        content.append(f"Núcleos: {env.get('cpu_cores_logical', 'N/A')}")
        content.append(f"RAM: {env.get('ram_total_gb', 'N/A')} GB")
        content.append(f"SO: {env.get('os', platform.system())}")
    else:
        content.append(f"CPU: {platform.processor()}")
        content.append(f"Sistema: {platform.system()} {platform.release()}")
    
    # Render text
    text_content = '\n'.join(content)
    ax_main.text(0.02, 0.98, text_content, transform=ax_main.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 linespacing=1.5)
    
    # Footer
    ax_footer = fig.add_axes([0, 0, 1, 0.08])
    ax_footer.set_facecolor(COLORS['background'])
    ax_footer.axis('off')
    ax_footer.text(0.5, 0.5, 'Sistema de Benchmark v2.1',
                   ha='center', va='center', fontsize=8, color='gray', style='italic')
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# PÁGINA 2: ALGORITMO CGNR
# =============================================================================
def create_algorithm_page(pdf: PdfPages):
    """Create algorithm explanation page."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.88])
    ax.axis('off')
    
    content = """
                    ALGORITMO CGNR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PROBLEMA DE RECONSTRUÇÃO
──────────────────────────────────────────────────────

A reconstrução tomográfica resolve o sistema linear:

                    g = H·f + n

Onde:
  g ∈ ℝᵐ : Medições (sinais recebidos)
  H ∈ ℝᵐˣⁿ : Matriz de sistema
  f ∈ ℝⁿ : Imagem a reconstruir
  n ∈ ℝᵐ : Ruído


2. ALGORITMO CGNR
──────────────────────────────────────────────────────

O CGNR minimiza ‖Hf - g‖₂² via gradiente conjugado:

  1. r₀ = Hᵀ(g - Hf₀)
  2. p₀ = r₀
  3. Para k = 0, 1, 2, ...
     αₖ = ‖rₖ‖² / ‖Hpₖ‖²
     fₖ₊₁ = fₖ + αₖpₖ
     rₖ₊₁ = rₖ - αₖHᵀHpₖ
     βₖ = ‖rₖ₊₁‖² / ‖rₖ‖²
     pₖ₊₁ = rₖ₊₁ + βₖpₖ

Critério de parada: ‖rₖ‖/‖r₀‖ < ε ou k > max_iter


3. MÉTRICAS
──────────────────────────────────────────────────────

• Tempo de Solver: Tempo do algoritmo CGNR
• Latência: Tempo total (fila + solver)
• Erro Final: ‖Hf - g‖₂ / ‖g‖₂
• Epsilon: ‖rₖ‖ / ‖r₀‖
• Speedup: T_python / T_cpp

INTERPRETAÇÃO DO ERRO FINAL:
  < 0.1 (10%): Excelente convergência
  0.1-0.5: Convergência aceitável
  0.5-1.0: Convergência parcial
  > 1.0: Divergência (solução inválida)
"""
    
    ax.text(0.0, 1.0, content, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            linespacing=1.3)
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# PÁGINA 3: COMPARATIVO PYTHON VS C++ (GIL, JIT, OpenMP)
# =============================================================================
def create_comparison_page(pdf: PdfPages):
    """Create Python vs C++ comparison with GIL, JIT, OpenMP explanations."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.88])
    ax.axis('off')
    
    content = """
         PYTHON vs C++: ANÁLISE TÉCNICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GLOBAL INTERPRETER LOCK (GIL) - Python
──────────────────────────────────────────────────────

O GIL é um mutex que protege objetos Python, impedindo
que múltiplas threads executem bytecode simultaneamente.

IMPACTO NO BENCHMARK:
• Python NÃO consegue usar múltiplos cores em código
  Python puro, mesmo com threading
• NumPy libera o GIL durante operações matriciais,
  mas há overhead na aquisição/liberação
• Jobs Python competem pelo GIL, causando contenção


2. JIT vs COMPILAÇÃO
─────────────────────────────────────────────────

• NumPy: Pré-compilado C/Fortran (NÃO é JIT)
• Numba: JIT real (não usado neste benchmark)
• C++: Compilado nativamente com otimizações


3. OpenMP (C++)
─────────────────────────────────────────────────

Paralelismo real via #pragma:
• H·p (produto matriz-vetor) → threads dividem
• Hᵀ·r (transposta) → idem
• Normas → redução paralela


4. TABELA COMPARATIVA
─────────────────────────────────────────────────

 Aspecto        | Python         | C++
 ───────────────┼────────────────┼────────────────
 Execução       | Interpretada   | Compilada
 Threads        | Bloqueadas GIL | OpenMP real
 Cache          | Memória Python | Binário .bin
 I/O            | Parse CSV      | Load binário


5. PORQUE C++ É MAIS RÁPIDO
─────────────────────────────────────────────────

1. Sem GIL → Threads reais
2. OpenMP → Loops paralelos
3. Eigen → SIMD vetorizado
4. Cache binário → Zero parsing
5. Memória contígua → Menos alocações
"""
    
    ax.text(0.0, 1.0, content, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            linespacing=1.35)
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# PÁGINA 4: ARQUITETURA DO EXPERIMENTO
# =============================================================================
def create_architecture_page(pdf: PdfPages):
    """Create experiment architecture explanation."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.88])
    ax.axis('off')
    
    content = """
              ARQUITETURA DO EXPERIMENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SERVIDORES INDEPENDENTES
──────────────────────────────────────────────────────

Python e C++ rodam como SERVIDORES SEPARADOS:

  ┌──────────────┐     HTTP     ┌──────────────┐
  │   CLIENTE    │ ───────────► │ Python:8081  │
  │  (gerador)   │              └──────────────┘
  │              │     HTTP     ┌──────────────┐
  │              │ ───────────► │  C++:8082    │
  └──────────────┘              └──────────────┘

• NÃO compartilham CPU (processos separados)
• NÃO compartilham memória
• Competem apenas por I/O e cache L3


2. TIMELINE DE UM JOB
──────────────────────────────────────────────────────

  Chegada        Início Exec         Fim
     │               │                │
     ▼               ▼                ▼
     ├───────────────┼────────────────┤
     │     FILA      │    EXECUÇÃO    │
     │   (cinza)     │   (colorido)   │
     └───────────────┴────────────────┘

     ◄── Latência Total ──────────────►

NOTA: Os "buracos" no Gantt (espaços sem cor) indicam:
  - Intervalos entre jobs (nenhum job ativo)
  - Tempo de rede/serialização
  - Overhead de I/O


3. PROFUNDIDADE DA FILA
──────────────────────────────────────────────────────

O gráfico "Profundidade da Fila" mostra:
  Y = Número de jobs aguardando processamento

INTERPRETAÇÃO:
  • Fila alta → Servidor sobrecarregado
  • Fila zero → Servidor ocioso
  • Fila estável → Throughput equilibrado

Se Python tem fila maior que C++:
  → Python está mais lento para drenar jobs


4. POR QUE GANTT SEPARADOS?
──────────────────────────────────────────────────────

Os Gantts são separados por servidor porque:
  • Não competem por CPU diretamente
  • Cada servidor processa sua própria fila
  • Comparação lado-a-lado é mais clara

O gráfico COMBINADO de fila mostra a competição
indireta por recursos do sistema (RAM, cache, I/O).
"""
    
    ax.text(0.0, 1.0, content, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            linespacing=1.25)
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# GRÁFICO DE GANTT COMBINADO
# =============================================================================
def create_combined_gantt(df: pd.DataFrame, pdf: PdfPages, input_dir: Path):
    """Create combined Gantt chart showing both servers."""
    if df.empty:
        return
    
    df = preprocess_jobs(df)
    
    fig = plt.figure(figsize=(11, 9))
    fig.suptitle('Diagrama de Gantt Combinado - Python vs C++', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    ax = fig.add_axes([0.15, 0.15, 0.78, 0.75])
    
    # Sort by arrival globally
    df = df.sort_values('t_arrival').reset_index(drop=True)
    
    n_jobs = len(df)
    bar_height = 0.7
    
    for idx, (_, row) in enumerate(df.iterrows()):
        y = n_jobs - idx - 1  # Invert Y so first job is at top
        server = row['server']
        color = COLORS[server]
        
        # 1. Queue time (gray)
        if row['queue_duration'] > 0.01:  # Only show if > 10ms
            ax.barh(y, row['queue_duration'], left=row['t_arrival'],
                    height=bar_height, color=COLORS['queue'], 
                    edgecolor='black', linewidth=0.3)
        
        # 2. Execution time (colored)
        ax.barh(y, row['exec_duration'], left=row['t_exec_start'],
                height=bar_height, color=color,
                edgecolor='black', linewidth=0.5)
        
        # 3. Arrival marker
        ax.plot([row['t_arrival']], [y], marker='|', color=COLORS['arrival'], 
                markersize=10, markeredgewidth=2)
    
    # Labels
    job_labels = [f"{row['job_id'][:10]} [{row['server'][0].upper()}]" 
                  for _, row in df.iterrows()]
    ax.set_yticks(range(n_jobs))
    ax.set_yticklabels(job_labels[::-1], fontsize=7)
    ax.set_xlabel('Tempo (segundos desde T₀)', fontsize=11)
    ax.set_ylabel('Jobs (ordenados por chegada)', fontsize=11)
    
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_xlim(0, df['t_end'].max() * 1.05)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['queue'], edgecolor='black', 
                      label='Tempo na Fila', linewidth=0.5),
        mpatches.Patch(facecolor=COLORS['python'], edgecolor='black',
                      label='Execução Python', linewidth=0.5),
        mpatches.Patch(facecolor=COLORS['cpp'], edgecolor='black',
                      label='Execução C++', linewidth=0.5),
        Line2D([0], [0], color=COLORS['arrival'], marker='|', linestyle='None',
               markersize=10, label='Chegada'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add explanation text box
    explanation = (
        "Leitura do Gráfico:\n"
        "• Barras CINZA = Job aguardando na fila\n"
        "• Barras AZUL/VERMELHA = Job em execução\n"
        "• Espaços em branco = Nenhum job naquele instante\n"
        "• [P] = Python, [C] = C++"
    )
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    graphs_dir = input_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    fig.savefig(graphs_dir / 'gantt_combined.png', dpi=300, bbox_inches='tight')
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# GRÁFICOS DE GANTT POR SERVIDOR
# =============================================================================
def create_server_gantt(df: pd.DataFrame, pdf: PdfPages, input_dir: Path):
    """Create per-server Gantt charts."""
    if df.empty:
        return
    
    df = preprocess_jobs(df)
    
    for server in ['cpp', 'python']:
        server_df = df[df['server'] == server].copy()
        if server_df.empty:
            continue
        
        server_df = server_df.sort_values('t_arrival').reset_index(drop=True)
        
        fig = plt.figure(figsize=(11, 7))
        fig.suptitle(f'Diagrama de Gantt - {server.upper()}',
                     fontsize=14, fontweight='bold', y=0.98)
        
        ax = fig.add_axes([0.18, 0.15, 0.75, 0.75])
        
        n_jobs = len(server_df)
        bar_height = 0.6
        
        for idx, (_, row) in enumerate(server_df.iterrows()):
            y = idx
            
            # Queue time
            if row['queue_duration'] > 0.01:
                ax.barh(y, row['queue_duration'], left=row['t_arrival'],
                        height=bar_height, color=COLORS['queue'], 
                        edgecolor='black', linewidth=0.3)
            
            # Execution time
            ax.barh(y, row['exec_duration'], left=row['t_exec_start'],
                    height=bar_height, color=COLORS[server],
                    edgecolor='black', linewidth=0.5)
            
            # Arrival marker
            ax.annotate('', xy=(row['t_arrival'], y + 0.4),
                        xytext=(row['t_arrival'], y + 0.7),
                        arrowprops=dict(arrowstyle='->', color=COLORS['arrival'], lw=1.2))
        
        job_labels = [f"{row['job_id']}\n{row['dataset_id']}" 
                      for _, row in server_df.iterrows()]
        ax.set_yticks(range(n_jobs))
        ax.set_yticklabels(job_labels, fontsize=7)
        ax.set_xlabel('Tempo (s)', fontsize=11)
        ax.set_ylabel('Jobs', fontsize=11)
        
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_xlim(0, df['t_end'].max() * 1.05)
        
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['queue'], label='Fila'),
            mpatches.Patch(facecolor=COLORS[server], label='Execução'),
            Line2D([0], [0], color=COLORS['arrival'], marker='v', linestyle='None',
                   markersize=8, label='Chegada'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        graphs_dir = input_dir / 'graphs'
        graphs_dir.mkdir(exist_ok=True)
        fig.savefig(graphs_dir / f'gantt_{server}.png', dpi=300, bbox_inches='tight')
        
        pdf.savefig(fig)
        plt.close(fig)


# =============================================================================
# ANÁLISE DE FILA
# =============================================================================
def create_queue_analysis(df: pd.DataFrame, pdf: PdfPages, input_dir: Path):
    """Create queue depth visualization with explanations."""
    if df.empty:
        return
    
    df = preprocess_jobs(df)
    
    fig = plt.figure(figsize=(11, 9))
    fig.suptitle('Análise da Fila de Processamento', fontsize=14, fontweight='bold')
    
    t_max = df['t_end'].max()
    time_points = np.linspace(0, t_max, 500)
    
    queue_data = {'time': time_points}
    
    for server in ['python', 'cpp']:
        server_df = df[df['server'] == server]
        queue_depth = []
        executing = []
        
        for t in time_points:
            arrived = server_df['t_arrival'] <= t
            not_finished = server_df['t_end'] > t
            in_system = arrived & not_finished
            
            is_executing = (server_df['t_exec_start'] <= t) & (server_df['t_end'] > t)
            
            n_in_system = in_system.sum()
            n_executing = is_executing.sum()
            n_in_queue = max(0, n_in_system - n_executing)
            
            queue_depth.append(n_in_queue)
            executing.append(n_executing)
        
        queue_data[f'{server}_queue'] = queue_depth
        queue_data[f'{server}_exec'] = executing
    
    # Plot 1: Queue depth
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.fill_between(time_points, queue_data['python_queue'], alpha=0.4, 
                     color=COLORS['python'], label='Python')
    ax1.fill_between(time_points, queue_data['cpp_queue'], alpha=0.4,
                     color=COLORS['cpp'], label='C++')
    ax1.plot(time_points, queue_data['python_queue'], color=COLORS['python'], linewidth=2)
    ax1.plot(time_points, queue_data['cpp_queue'], color=COLORS['cpp'], linewidth=2)
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Jobs Aguardando')
    ax1.set_title('Profundidade da Fila = Quantos jobs estão ESPERANDO para processar', 
                  fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add interpretation
    ax1.text(0.02, 0.95, 
             "Interpretação: Linha ALTA = Servidor sobrecarregado, jobs acumulando",
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Jobs executing
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.fill_between(time_points, queue_data['python_exec'], alpha=0.4,
                     color=COLORS['python'], label='Python')
    ax2.fill_between(time_points, queue_data['cpp_exec'], alpha=0.4,
                     color=COLORS['cpp'], label='C++')
    ax2.plot(time_points, queue_data['python_exec'], color=COLORS['python'], linewidth=2)
    ax2.plot(time_points, queue_data['cpp_exec'], color=COLORS['cpp'], linewidth=2)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Jobs Executando')
    ax2.set_title('Jobs em Execução = Quantos jobs estão PROCESSANDO agora', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    ax2.text(0.02, 0.95,
             "Interpretação: Valor > 0 = Servidor ocupado processando",
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    graphs_dir = input_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    fig.savefig(graphs_dir / 'queue_analysis.png', dpi=300, bbox_inches='tight')
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# ANÁLISE ESTATÍSTICA
# =============================================================================
def create_statistical_analysis(df: pd.DataFrame, pdf: PdfPages, input_dir: Path):
    """Create statistical analysis page."""
    if df.empty:
        return
    
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle('Análise Estatística de Desempenho', fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    palette = {'python': COLORS['python'], 'cpp': COLORS['cpp']}
    
    # Plot 1: Solver Time
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df, x='dataset_id', y='solver_time_ms', hue='server',
                palette=palette, ax=ax1)
    ax1.set_ylabel('Tempo de Solver (ms)')
    ax1.set_xlabel('Dataset')
    ax1.set_title('Distribuição do Tempo de Solver')
    ax1.legend(title='Servidor', fontsize=8)
    
    # Plot 2: Latency
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df, x='dataset_id', y='latency_ms', hue='server',
                palette=palette, ax=ax2)
    ax2.set_ylabel('Latência Total (ms)')
    ax2.set_xlabel('Dataset')
    ax2.set_title('Distribuição da Latência')
    ax2.legend(title='Servidor', fontsize=8)
    
    # Plot 3: Speedup
    ax3 = fig.add_subplot(gs[1, 0])
    speedup_data = []
    for ds in df['dataset_id'].unique():
        py_mean = df[(df['dataset_id'] == ds) & (df['server'] == 'python')]['solver_time_ms'].mean()
        cpp_mean = df[(df['dataset_id'] == ds) & (df['server'] == 'cpp')]['solver_time_ms'].mean()
        speedup = py_mean / cpp_mean if cpp_mean > 0 else 0
        speedup_data.append({'dataset_id': ds, 'speedup': speedup})
    
    speedup_df = pd.DataFrame(speedup_data)
    bars = ax3.bar(speedup_df['dataset_id'], speedup_df['speedup'], 
                   color=[COLORS['cpp'] if s > 1 else COLORS['python'] for s in speedup_df['speedup']],
                   edgecolor='black')
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Igual (1x)')
    ax3.set_ylabel('Speedup')
    ax3.set_xlabel('Dataset')
    ax3.set_title('Speedup (C++ vs Python)')
    
    for bar, val in zip(bars, speedup_df['speedup']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}x', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 4: Error
    ax4 = fig.add_subplot(gs[1, 1])
    for server, color in [('python', COLORS['python']), ('cpp', COLORS['cpp'])]:
        data = df[df['server'] == server]['final_error']
        ax4.hist(data, bins=15, alpha=0.6, color=color, label=server.upper(), edgecolor='black')
    ax4.set_xlabel('Erro Final')
    ax4.set_ylabel('Frequência')
    ax4.set_title('Distribuição do Erro Final')
    ax4.legend()
    ax4.axvline(x=0.5, color='red', linestyle='--', label='Limite aceitável')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    graphs_dir = input_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    fig.savefig(graphs_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# GALERIA DE IMAGENS - UMA IMAGEM POR PÁGINA, ORDENADAS
# =============================================================================
def create_image_gallery(input_dir: Path, pdf: PdfPages):
    """Create gallery with 1 representative image per dataset+server, on separate pages."""
    img_dir = input_dir / 'images'
    if not img_dir.exists():
        return
    
    # Find all meta files
    meta_files = sorted(list(img_dir.glob('*_meta.json')))
    if not meta_files:
        return
    
    # Collect one representative image per (dataset_id, server) combination
    unique_images = {}  # key: (dataset_id, server) -> value: entry
    
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            csv_path = meta_path.parent / meta_path.name.replace('_meta.json', '_image.csv')
            if csv_path.exists():
                dataset_id = meta.get('dataset_id', 'unknown')
                server = meta.get('server', 'unknown').lower()
                key = (dataset_id, server)
                
                # Keep the first (or best by error) image per combination
                if key not in unique_images:
                    unique_images[key] = {'csv': csv_path, 'meta': meta}
                else:
                    # Keep the one with lower error
                    current_err = unique_images[key]['meta'].get('final_error', float('inf'))
                    new_err = meta.get('final_error', float('inf'))
                    if new_err < current_err:
                        unique_images[key] = {'csv': csv_path, 'meta': meta}
        except:
            pass
    
    if not unique_images:
        return
    
    # Sort by (dataset_id, server) for consistent ordering
    sorted_keys = sorted(unique_images.keys())
    selected_images = [unique_images[k] for k in sorted_keys]
    
    print(f"[INFO] Image gallery: {len(selected_images)} unique dataset+server combinations")
    
    # Create one page per image
    for idx, entry in enumerate(selected_images):
        fig = plt.figure(figsize=(11, 9))
        
        meta = entry['meta']
        dataset_id = meta.get('dataset_id', '?')
        server = meta.get('server', '?').upper()
        err = meta.get('final_error', 0)
        iterations = meta.get('iterations', 0)
        solver_time = meta.get('solver_time_ms', 0)
        converged = meta.get('converged', False)
        
        title = f"Imagem Reconstruída: {dataset_id} [{server}]"
        title += f"\nPágina {idx + 1}/{len(selected_images)}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        ax = fig.add_axes([0.1, 0.2, 0.75, 0.65])  # Main image
        
        try:
            data = np.loadtxt(entry['csv'], delimiter=',')
            
            # Check for divergence
            is_diverged = False
            if not np.isfinite(data).all() or np.max(np.abs(data)) > 1e6:
                is_diverged = True
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Use min/max normalization for best contrast
            v_min = np.min(data)
            v_max = np.max(data)
            if v_max <= v_min:
                v_max = v_min + 1.0
            
            im = ax.imshow(data, cmap='inferno', vmin=v_min, vmax=v_max, 
                          aspect='equal', interpolation='nearest')
            
            # Colorbar
            cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.65])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label('Intensidade', fontsize=10)
            
            ax.axis('off')
            
            # Info box
            status = "DIVERGIU" if is_diverged else ("Convergiu" if converged else "Max Iterações")
            info_text = (
                f"Dataset: {dataset_id}\n"
                f"Servidor: {server}\n"
                f"Dimensão: {data.shape[0]}x{data.shape[1]}\n"
                f"Iterações: {iterations}\n"
                f"Tempo: {solver_time:.1f} ms\n"
                f"Erro Final: {err:.4e}\n"
                f"Status: {status}"
            )
            
            ax_info = fig.add_axes([0.1, 0.02, 0.8, 0.12])
            ax_info.axis('off')
            ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                        fontsize=11, fontfamily='monospace',
                        verticalalignment='center', horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Erro ao carregar:\n{str(e)[:50]}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # Save first page as PNG
        if idx == 0:
            graphs_dir = input_dir / 'graphs'
            graphs_dir.mkdir(exist_ok=True)
            fig.savefig(graphs_dir / 'image_gallery.png', dpi=300, bbox_inches='tight')
        
        pdf.savefig(fig)
        plt.close(fig)


# =============================================================================
# TABELA ESTATÍSTICA
# =============================================================================
def create_stats_table(df: pd.DataFrame, pdf: PdfPages):
    """Create statistics summary table page."""
    if df.empty:
        return
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.85])
    ax.axis('off')
    
    # Build table data
    lines = []
    lines.append("TABELA ESTATÍSTICA DETALHADA")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Métrica':<20} {'Servidor':<10} {'Média':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    lines.append("-" * 80)
    
    for metric, name in [('solver_time_ms', 'Solver (ms)'), 
                         ('latency_ms', 'Latência (ms)'),
                         ('final_error', 'Erro Final')]:
        for server in ['python', 'cpp']:
            data = df[df['server'] == server][metric]
            if len(data) > 0:
                lines.append(
                    f"{name:<20} {server.upper():<10} "
                    f"{data.mean():>12.2f} {data.std():>12.2f} "
                    f"{data.min():>12.2f} {data.max():>12.2f}"
                )
        lines.append("")
    
    lines.append("=" * 80)
    
    # Summary
    py_mean = df[df['server'] == 'python']['solver_time_ms'].mean()
    cpp_mean = df[df['server'] == 'cpp']['solver_time_ms'].mean()
    speedup = py_mean / cpp_mean if cpp_mean > 0 else 0
    
    lines.append("")
    lines.append(f"SPEEDUP GLOBAL: {speedup:.2f}x")
    lines.append(f"(C++ é {speedup:.1f} vezes mais rápido que Python)")
    lines.append("")
    lines.append(f"Total de Jobs: {len(df)}")
    lines.append(f"  Python: {len(df[df['server'] == 'python'])}")
    lines.append(f"  C++:    {len(df[df['server'] == 'cpp'])}")
    
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# CONCLUSÕES
# =============================================================================
def create_conclusions_page(df: pd.DataFrame, pdf: PdfPages):
    """Create conclusions page."""
    if df.empty:
        return
    
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.88])
    ax.axis('off')
    
    py_mean = df[df['server'] == 'python']['solver_time_ms'].mean()
    cpp_mean = df[df['server'] == 'cpp']['solver_time_ms'].mean()
    speedup = py_mean / cpp_mean if cpp_mean > 0 else 0
    
    content = f"""
                        CONCLUSÕES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RESULTADOS PRINCIPAIS
──────────────────────────────────────────────────────

O servidor C++ demonstrou speedup de {speedup:.2f}x.

Tempos médios:
  Python: {py_mean:.2f} ms
  C++:    {cpp_mean:.2f} ms


2. PORQUE C++ FOI MAIS RÁPIDO
──────────────────────────────────────────────────────

• SEM GIL → Threads reais via OpenMP
• Eigen otimizado → Operações SIMD vetorizadas
• Cache binário → Zero parsing de CSV
• Compilação → Código nativo vs interpretado


3. ONDE OpenMP CONTRIBUIU
──────────────────────────────────────────────────────

OpenMP paralelizou as operações mais custosas:
• Multiplicação matriz-vetor (H·p)
• Multiplicação transposta (Hᵀ·r)
• Cálculo de normas (redução paralela)


4. LIMITAÇÕES
──────────────────────────────────────────────────────

• Testes em localhost (sem latência de rede real)
• Python sem Numba (JIT desabilitado)
• Número fixo de iterações


5. REFERÊNCIAS
──────────────────────────────────────────────────────

[1] Maziero, C. A. "Sistemas Operacionais"
    http://wiki.inf.ufpr.br/maziero/

[2] Eigen C++ - https://eigen.tuxfamily.org/

[3] OpenMP - https://www.openmp.org/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   FIM DO RELATÓRIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    ax.text(0.0, 1.0, content, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            linespacing=1.3)
    
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# MAIN GENERATOR
# =============================================================================
def generate_scientific_report(input_dir: Path, output_dir: Path):
    """Generate the complete academic-quality PDF report."""
    print(f"[INFO] Generating report for: {input_dir.name}")
    
    df_jobs, df_sys, env = load_data(input_dir)
    if df_jobs is None:
        print("[ERROR] No telemetry data found.")
        return False
    
    print(f"[INFO] Loaded {len(df_jobs)} job records")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'graphs').mkdir(exist_ok=True)
    
    pdf_path = output_dir / "Relatorio_Cientifico_Final.pdf"
    print(f"[INFO] Creating PDF: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        print("[INFO] Page 1: Cover...")
        create_cover_page(df_jobs, env, pdf, input_dir)
        
        print("[INFO] Page 2: Algorithm...")
        create_algorithm_page(pdf)
        
        print("[INFO] Page 3: Python vs C++ (GIL, JIT, OpenMP)...")
        create_comparison_page(pdf)
        
        print("[INFO] Page 4: Architecture...")
        create_architecture_page(pdf)
        
        print("[INFO] Page 5: Combined Gantt...")
        create_combined_gantt(df_jobs, pdf, input_dir)
        
        print("[INFO] Pages 6-7: Per-server Gantt...")
        create_server_gantt(df_jobs, pdf, input_dir)
        
        print("[INFO] Page 8: Queue analysis...")
        create_queue_analysis(df_jobs, pdf, input_dir)
        
        print("[INFO] Page 9: Statistics...")
        create_statistical_analysis(df_jobs, pdf, input_dir)
        
        print("[INFO] Page 10: Stats table...")
        create_stats_table(df_jobs, pdf)
        
        print("[INFO] Page 11: Image gallery...")
        create_image_gallery(input_dir, pdf)
        
        print("[INFO] Page 12: Conclusions...")
        create_conclusions_page(df_jobs, pdf)
    
    print(f"[SUCCESS] PDF generated: {pdf_path}")
    return True


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PDF report')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=False)
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else input_path
    
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)
    
    success = generate_scientific_report(input_path, output_path)
    sys.exit(0 if success else 1)
