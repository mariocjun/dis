import imageio
import numpy as np
from pathlib import Path
import argparse

def resample_frames(frames, target_count):
    """Reamostra uma lista de frames para um número alvo de frames."""
    if not frames:
        return []

    resampled = []
    ratio = len(frames) / target_count
    for i in range(target_count):
        original_index = int(i * ratio)
        # Garante que o índice não saia dos limites
        original_index = min(original_index, len(frames) - 1)
        resampled.append(frames[original_index])
    return resampled

def combine_animations(precond_path, standard_path, output_path, duration_per_gif=10, fps=10):
    """Combina duas animações GIF em uma só, reamostrando os frames para que cada parte dure o tempo especificado."""
    try:
        # Carrega os dois GIFs
        reader_precond = imageio.get_reader(precond_path)
        reader_standard = imageio.get_reader(standard_path)

        # Pega os frames de cada GIF
        frames_precond = [reader_precond.get_data(i) for i in range(len(reader_precond))]
        frames_standard = [reader_standard.get_data(i) for i in range(len(reader_standard))]

        if not frames_precond or not frames_standard:
            print("[ERRO] Um dos GIFs de entrada está vazio ou não pôde ser lido.")
            return

        # Calcula o número de frames necessários para cada parte para atingir a duração desejada com o FPS alvo
        target_frames_per_part = int(duration_per_gif * fps)

        # Reamostra os frames de cada animação
        resampled_precond = resample_frames(frames_precond, target_frames_per_part)
        resampled_standard = resample_frames(frames_standard, target_frames_per_part)

        # Concatena os frames reamostrados
        combined_frames = resampled_precond + resampled_standard

        # Salva o novo GIF com um FPS constante
        imageio.mimsave(output_path, combined_frames, fps=fps)
        print(f"[SUCESSO] Animação combinada salva em: {output_path}")

    except FileNotFoundError as e:
        print(f"[ERRO] Arquivo não encontrado: {e}. Verifique os caminhos.")
    except Exception as e:
        print(f"[ERRO] Falha ao combinar animações: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Combina as animações 'precond' e 'standard' em um único GIF."
    )
    parser.add_argument(
        "animations_folder",
        type=str,
        help="Caminho para a pasta contendo as animações (ex: output_csv/animations)."
    )
    parser.add_argument(
        "base_name",
        type=str,
        help="Nome base dos arquivos a serem combinados (ex: image_30x30_g1_sparse)."
    )
    args = parser.parse_args()

    animations_folder = Path(args.animations_folder)
    precond_gif = animations_folder / f"{args.base_name}_precond_animation.gif"
    standard_gif = animations_folder / f"{args.base_name}_standard_animation.gif"
    combined_gif = animations_folder / f"{args.base_name}_combined_animation.gif"

    print("--- Combinador de Animações ---")
    combine_animations(precond_gif, standard_gif, combined_gif)

if __name__ == "__main__":
    main()
