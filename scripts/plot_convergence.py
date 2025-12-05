import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(metrics_folder, base_names):
    """Plota as curvas de convergência para um ou mais resultados."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    if not base_names:
        # Se nenhum nome base for fornecido, encontra todos os arquivos de histórico
        search_pattern = os.path.join(metrics_folder, "convergence_history_*.csv")
        all_files = glob.glob(search_pattern)
        if not all_files:
            print("[AVISO] Nenhum arquivo 'convergence_history_*.csv' encontrado na pasta de métricas.")
            return

        # Extrai os nomes dos testes para a legenda
        labels = ['_'.join(Path(f).stem.split('_')[2:]) for f in all_files]
        files_to_plot = all_files
    else:
        # Monta os caminhos dos arquivos a partir dos nomes base
        files_to_plot = [os.path.join(metrics_folder, f"convergence_history_{name}.csv") for name in base_names]
        labels = base_names

    for i, file_path in enumerate(files_to_plot):
        try:
            # Carrega os dados: Iteration,ResidualNorm,SolutionNorm,ExecutionTime_ms
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            iterations = data[:, 0]
            residual_norm = data[:, 1]

            # Usa o nome do arquivo (sem prefixo/sufixo) como legenda
            label = labels[i]
            ax.plot(iterations, residual_norm, marker='o', linestyle='-', markersize=4, label=label)

        except Exception as e:
            print(f"[AVISO] Não foi possível processar o arquivo '{file_path}': {e}")
            continue

    ax.set_yscale('log')
    ax.set_title('Curva de Convergência (Norma do Resíduo)', fontsize=16)
    ax.set_xlabel('Iteração', fontsize=12)
    ax.set_ylabel('Norma do Resíduo (log)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--")

    # Salva o gráfico
    output_path = Path(metrics_folder).parent / "convergence_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\n[SUCESSO] Gráfico de convergência salvo em: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plota as curvas de convergência a partir dos arquivos de histórico gerados pelo benchmark."
    )
    parser.add_argument(
        "metrics_folder",
        type=str,
        help="Caminho para a pasta contendo os arquivos CSV de métricas (ex: ../output_csv/metrics)."
    )
    parser.add_argument(
        "base_name",
        nargs='*',
        help="Nomes base dos resultados a serem plotados (ex: 30x30_g1_sparse_standard). Se omitido, plota todos."
    )
    args = parser.parse_args()

    plot_convergence(args.metrics_folder, args.base_name)


if __name__ == "__main__":
    print("--- Plotter de Curva de Convergência ---")
    print("Requer: pip install matplotlib numpy")
    main()

