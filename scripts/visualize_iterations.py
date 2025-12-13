import argparse
import glob
import os
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np


def create_animation(image_folder, base_name, output_folder):
    """Gera um GIF animado a partir de uma sequência de imagens CSV."""
    print(f"\nProcessando: {base_name}")
    search_pattern = f"{base_name}_iter_*.csv"
    file_paths = sorted(
        glob.glob(os.path.join(image_folder, search_pattern)),
        key=lambda x: int(Path(x).stem.split('_')[-1])
    )

    if not file_paths:
        print(f"  [AVISO] Nenhum arquivo de imagem encontrado para '{base_name}'. Pulando.")
        return

    images = []
    solver_type = base_name.split('_')[-1]  # Extrai 'precond' ou 'standard'

    for i, file_path in enumerate(file_paths):
        try:
            # Carrega a imagem CSV e a remodela
            img_data = np.loadtxt(file_path, delimiter=',')

            # Cria o plot
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.suptitle(solver_type.capitalize(), fontsize=16)
            im = ax.imshow(img_data, cmap='viridis', vmin=np.min(img_data), vmax=np.max(img_data))
            ax.set_title(f"Iteração {i}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            # Adiciona uma colorbar para referência de intensidade
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

            # --- NOVO: Salva o frame individual como PNG ---
            frames_dir = Path(output_folder).parent / "frames" / base_name
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / f"iter_{i}.png"
            fig.savefig(frame_path, dpi=100) # Removed bbox_inches='tight' for consistency
            # -----------------------------------------------

            # Converte o plot para imagem para o GIF usando o mesmo output do savefig
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100) # Removed bbox_inches='tight' for consistency
            buf.seek(0)
            image = imageio.imread(buf)
            images.append(image)

            plt.close(fig)

        except Exception as e:
            print(f"  [ERRO] Falha ao processar {file_path}: {e}")
            continue

    if not images:
        print(f"  [ERRO] Nenhuma imagem pôde ser gerada para '{base_name}'.")
        return

    # Salva o GIF
    output_path = os.path.join(output_folder, f"{base_name}_animation.gif")
    print(f"  [DEBUG] Saving GIF with {len(images)} frames to {output_path}")
    try:
        imageio.mimsave(output_path, images, fps=5)
        print(f"  [SUCESSO] Animação salva em: {output_path}")
    except Exception as e:
        print(f"  [ERRO] Falha ao salvar o GIF em {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Cria animações GIF a partir dos CSVs de iteração gerados pelo benchmark."
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Caminho para a pasta contendo os arquivos CSV das imagens (ex: ../output_csv/images)."
    )
    parser.add_argument(
        "base_name",
        nargs='*',
        help="Nomes base para agrupar as imagens (ex: image_30x30_g1_sparse_standard). Se omitido, processa todos."
    )
    args = parser.parse_args()

    output_folder = Path(args.image_folder).parent / "animations"
    output_folder.mkdir(exist_ok=True)
    print(f"Pasta de saída para animações: {output_folder}")

    if not args.base_name:
        # Encontra todos os prefixos únicos na pasta
        all_files = glob.glob(os.path.join(args.image_folder, "*_iter_*.csv"))
        base_names = sorted(list(set('_'.join(Path(f).stem.split('_')[:-2]) for f in all_files)))
        if not base_names:
            print("[AVISO] Nenhum arquivo de iteração encontrado na pasta especificada.")
            return
    else:
        base_names = args.base_name

    for name in base_names:
        create_animation(args.image_folder, name, output_folder)


if __name__ == "__main__":
    print("--- Gerador de Animação de Iterações ---")
    print("Requer: pip install imageio matplotlib numpy")
    main()
