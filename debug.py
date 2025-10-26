# python
import os
import sys
import inspect

def _caller_dir():
    # percorre a pilha e retorna a primeira filename diferente deste arquivo
    this = os.path.abspath(__file__)
    for frame_info in inspect.stack()[1:]:
        fn = os.path.abspath(frame_info.filename)
        if fn != this:
            return os.path.dirname(fn)
    # fallback
    return os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()

def debug_files(which='module'):
    """
    which: 'module' -> pasta deste módulo (`debug.py`)
           'caller' -> pasta do arquivo que chamou esta função
           'cwd'    -> diretório de trabalho atual
    """
    if which == 'module':
        script_dir = os.path.dirname(os.path.abspath(__file__))
    elif which == 'caller':
        script_dir = _caller_dir()
    elif which == 'cwd':
        script_dir = os.getcwd()
    else:
        raise ValueError("which must be 'module', 'caller' or 'cwd'")

    print(f"Rodando a partir da pasta: {script_dir}")
    print("Arquivos:")
    for f in os.listdir(script_dir):
        print(" -", f)


if __name__ == "__main__":
    debug_files('module')
