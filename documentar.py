#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

DOC_DIR = "doc"
BASE_OUTPUT = "DOC"
EXT_ORDER = ["txt", "cpp", "hpp", "py", "yaml", "csv", "sh"]
MAX_CSV_SIZE = 10 * 1024

IGNORE_DIRS = ["cmake", ".venv", "__pycache__", "README", "doc"]
IGNORE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ico", ".svg"]

FENCE = chr(96) + chr(96) + chr(96)


def should_ignore_dir(dirname):
    for pattern in IGNORE_DIRS:
        if dirname.startswith(pattern):
            return True
    return False


def should_ignore_file(filepath):
    if any(filepath.suffix.lower() == ext for ext in IGNORE_EXTENSIONS):
        return True
    if filepath.suffix.lower() == ".csv":
        try:
            if filepath.stat().st_size > MAX_CSV_SIZE:
                return True
        except:
            return True
    return False


def get_valid_files(root_dir="."):
    valid_files = []
    root_path = Path(root_dir)
    for item in root_path.rglob("*"):
        if any(should_ignore_dir(part) for part in item.parts):
            continue
        if not item.is_file():
            continue
        ext = item.suffix.lower().lstrip(".")
        if ext not in EXT_ORDER:
            continue
        if should_ignore_file(item):
            continue
        valid_files.append(item)
    return valid_files


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def generate_tree(files):
    tree_items = {}
    file_sizes = {}
    root_name = Path.cwd().name
    tree_items["ROOT"] = (root_name + "/", -1, True, "")

    for filepath in files:
        parts = list(filepath.parts)
        if parts[0] == '.':
            parts = parts[1:]

        file_path_str = "/".join(parts)
        try:
            file_sizes[file_path_str] = filepath.stat().st_size
        except:
            file_sizes[file_path_str] = 0

        current_path_parts = []
        for i in range(len(parts)):
            current_path_parts.append(parts[i])
            path = "/".join(current_path_parts)

            if path not in tree_items:
                is_dir = i < len(parts) - 1
                parent_path = "/".join(current_path_parts[:-1]) if i > 0 else ""
                tree_items[path] = (parts[i], i, is_dir, parent_path)

    def sort_key(item):
        path_key = item[0]
        if path_key == "ROOT":
            return ()

        name, depth, is_dir_final, parent_path = item[1]

        parts = path_key.split('/')
        sort_tuple = []
        for i, part_name in enumerate(parts):
            is_part_dir = (i < len(parts) - 1) or is_dir_final
            sort_tuple.append((not is_part_dir, part_name.lower()))

        return tuple(sort_tuple)

    sorted_items = sorted(tree_items.items(), key=sort_key)

    lines = []

    last_children_map = {}
    for path, (name, depth, is_dir, parent_path) in sorted_items:
        if depth == -1: continue
        last_children_map[parent_path] = path

    prefixes_map = {"": ""}

    for path, (name, depth, is_dir, parent_path) in sorted_items:
        if depth == -1:
            lines.append(name)
            continue

        parent_prefix = prefixes_map[parent_path]
        is_last = (last_children_map.get(parent_path) == path)

        # --- INÍCIO DA MUDANÇA (CARACTERES 'PESADOS') ---
        if is_last:
            connector = "┗━━ "  # Símbolo 'pesado'
            child_prefix = parent_prefix + "    "
        else:
            connector = "┣━━ "  # Símbolo 'pesado'
            child_prefix = parent_prefix + "┃   "  # Símbolo 'pesado'
        # --- FIM DA MUDANÇA ---

        if is_dir:
            prefixes_map[path] = child_prefix
            display_name = name
        else:
            size = file_sizes.get(path, 0)
            display_name = f"{name} ({format_size(size)})"

        lines.append(f"{parent_prefix}{connector}{display_name}")

    return "\n".join(lines)


def get_language(ext):
    lang_map = {"txt": "text", "cpp": "cpp", "hpp": "cpp", "py": "python", "yaml": "yaml", "csv": "csv", "sh": "bash"}
    return lang_map.get(ext, "")


def generate_metadata(files, files_by_ext):
    now = datetime.now()
    total_size = sum(f.stat().st_size for f in files)
    metadata = f"""# Documentação do Projeto: {Path.cwd().name}

**Gerado em:** {now.strftime('%d/%m/%Y às %H:%M:%S')}  
**Total de arquivos:** {len(files)}  
**Tamanho total:** {format_size(total_size)}  
**Extensões incluídas:** {', '.join(EXT_ORDER)}

## Distribuição por tipo de arquivo

"""
    for ext in EXT_ORDER:
        if files_by_ext[ext]:
            ext_size = sum(f.stat().st_size for f in files_by_ext[ext])
            metadata += f"- **{ext.upper()}**: {len(files_by_ext[ext])} arquivo(s) - {format_size(ext_size)}\n"
    metadata += "\n---\n\n"
    return metadata


def main():
    doc_path = Path(DOC_DIR)
    doc_path.mkdir(exist_ok=True)
    now = datetime.now()
    filename = f"{BASE_OUTPUT}-{now.strftime('%d-%m-%Y--%H-%M')}.md"
    OUTPUT = doc_path / filename
    print(f"Arquivo de saída: {OUTPUT}")
    print("Coletando arquivos válidos...")
    files = get_valid_files()
    if not files:
        print("Nenhum arquivo válido encontrado!")
        return
    print(f"Encontrados {len(files)} arquivos válidos")
    files_by_ext = {}
    for ext in EXT_ORDER:
        files_by_ext[ext] = []
    for filepath in files:
        ext = filepath.suffix.lstrip(".").lower()
        if ext in files_by_ext:
            files_by_ext[ext].append(filepath)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        print("Gerando metadados...")
        f.write(generate_metadata(files, files_by_ext))
        print("Gerando estrutura de árvore...")
        f.write("# Estrutura do projeto:\n")
        f.write(FENCE + "text\n")
        f.write(generate_tree(files))
        f.write("\n" + FENCE + "\n\n")
        print("Gerando blocos de código...")
        f.write("---\n\n# Código dos Arquivos\n\n")
        for ext in EXT_ORDER:
            for filepath in sorted(files_by_ext[ext]):
                fname = filepath.name
                lang = get_language(ext)
                f.write(f"## arquivo {fname}\n")
                f.write(FENCE + lang + "\n")
                try:
                    content = filepath.read_text(encoding="utf-8")
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")
                except Exception as e:
                    f.write(f"# Erro ao ler arquivo: {e}\n")
                f.write(FENCE + "\n\n")
    print(f"\n✓ {filename} gerado com sucesso em {DOC_DIR}/")
    doc_files = sorted(doc_path.glob(f"{BASE_OUTPUT}-*.md"))
    if len(doc_files) > 1:
        print(f"\nVersões anteriores encontradas ({len(doc_files) - 1}):")
        for doc in doc_files[:-1]:
            size = format_size(doc.stat().st_size)
            mtime = datetime.fromtimestamp(doc.stat().st_mtime)
            print(f"  - {doc.name} ({size}) - {mtime.strftime('%d/%m/%Y %H:%M')}")


if __name__ == "__main__":
    main()
