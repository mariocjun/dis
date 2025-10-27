#!/bin/bash

# Verifica se o caminho do executável foi passado
if [ -z "$1" ]; then
  echo "Erro: Caminho do executável não fornecido."
  exit 1
fi

EXECUTABLE_PATH="$1"
# Pega todos os argumentos a partir do segundo (se houver)
shift # Remove o primeiro argumento (o caminho do executável)
PROGRAM_ARGS="$@" # Junta o resto

# Cria o nome do arquivo de log com data e hora
TIMESTAMP=$(date +"%d-%m-%Y--%H-%M-%S")
LOG_DIR="doc"
LOG_FILENAME="$LOG_DIR/CPL_RUN_$TIMESTAMP.md"

# Garante que o diretório 'doc' exista
mkdir -p "$LOG_DIR"

echo "--- Executando '$EXECUTABLE_PATH $PROGRAM_ARGS' ---"
echo "--- Log de execução será salvo em: '$LOG_FILENAME' ---"

# Executa o comando, redireciona stdout e stderr (2>&1),
# mostra no console (tee) e salva no arquivo.
# Usamos 'script' para capturar melhor a saída, incluindo cores se houver
script -q -c "$EXECUTABLE_PATH $PROGRAM_ARGS" /dev/null | tee "$LOG_FILENAME"

# Pega o código de saída do comando executado
# OBS: O código de saída capturado aqui pode ser do 'tee', não do executável diretamente
# Uma forma mais robusta seria executar sem o 'tee' primeiro, pegar o código, e depois rodar de novo com 'tee',
# mas isso executaria o programa duas vezes. Para logs, isso geralmente é suficiente.
EXIT_CODE=$?

echo "--- Execução concluída (Código de Saída: $EXIT_CODE) ---"

# Sai do script bash com o código de saída capturado
exit $EXIT_CODE