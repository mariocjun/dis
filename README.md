# DIS - Distributed Image System

Sistema de benchmark para reconstrução de imagem por ultrassom usando o algoritmo CGNR, comparando implementações em C++ e Python.

## Estrutura do Projeto

```
dis/
├── battle_orchestrator.py    # Orquestrador principal da batalha
├── battle_client.py          # Cliente para processamento de imagens
├── battle_graphs.py          # Gerador de gráficos
├── image_list_generator.py   # Gerador de lista de imagens sorteadas
├── resource_monitor.py       # Monitor de CPU, memória e HD
├── memory_limiter.py         # Limitador de memória por processo
├── server_python.py          # Servidor Python (CGNR)
├── main.cpp                  # Servidor C++ (CGNR)
├── data/                     # Matrizes H e sinais G
├── include/                  # Headers C++
├── src/                      # Fontes C++
├── battle_results/           # Resultados da batalha (gerado)
│   └── graphs/               # Gráficos gerados
└── old/                      # Arquivos antigos/não utilizados
```

## Execução

### Batalha Completa (3 clientes, 50 imagens cada)

```bash
python battle_orchestrator.py
```

### Opções

```bash
python battle_orchestrator.py --clients 3 --images 50 --memory-reduction 50
```

| Opção | Descrição | Default |
|-------|-----------|---------|
| `--clients` | Número de clientes simultâneos | 3 |
| `--images` | Imagens por cliente | 50 |
| `--memory-reduction` | % de redução do limite de memória | 50 |
| `--output` | Diretório de saída | battle_results |
| `--skip-servers` | Assume servidores já rodando | false |

### Apenas Gerar Gráficos

```bash
python battle_graphs.py --input battle_results
```

## Gráficos Gerados

- `memory_usage.png` - Uso de memória por cliente
- `cpu_usage.png` - Uso de CPU por cliente
- `io_usage.png` - I/O de disco
- `language_comparison.png` - C++ vs Python
- `dashboard.png` - Dashboard consolidado

## Dependências

```bash
pip install psutil matplotlib numpy
```

## Build C++

O orquestrador compila automaticamente, mas você pode compilar manualmente:

```bash
cmake -S . -B build
cmake --build build --config Release
```