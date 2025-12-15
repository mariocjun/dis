# 🔬 DIS - Distributed Imaging System
## Benchmark de Reconstrução de Imagens Ultrassônicas

Sistema de benchmark para comparação de desempenho entre implementações **C++** e **Python** do algoritmo CGNR (Conjugate Gradient Normal Residual) para reconstrução de imagens ultrassônicas.

---

## 📋 Pré-requisitos

### Windows
- **Visual Studio 2022+** (com componentes C++ Desktop Development)
- **CMake 3.20+**
- **Ninja** (instalado via Visual Studio ou separadamente)
- **Python 3.10+**

---

## 🔧 Compilação no Windows

### 1. Clone o repositório
```powershell
git clone <repo-url>
cd dis
```

### 2. Crie o ambiente virtual Python
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Compile o servidor C++

**Opção A: Via Developer Command Prompt (recomendado)**
```powershell
# Execute no Developer Command Prompt do Visual Studio
cmd /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake -S . -B build -G Ninja && cmake --build build --target UltrasoundServerHTTP --config Release"
```

**Opção B: Via CLion/VSCode**
- Abra o projeto
- Configure CMake com generator "Ninja"
- Build target: `UltrasoundServerHTTP`

### 4. Verifique a compilação
```powershell
# O executável estará em:
.\build\UltrasoundServerHTTP.exe
```

---

## 🚀 Execução do Benchmark

### Modo DEMO (Recomendado para começar)
```powershell
$env:PYTHONUTF8=1; python run_benchmark.py --demo
```

### Modo FULL (Teste completo com ganhos aleatórios)
```powershell
python run_benchmark.py --full
python run_benchmark.py --full --reps 20 --concurrency 6
```

---

## 📊 Como Funciona o Sistema

### Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                     run_benchmark.py                         │
│                    (Orquestrador)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Python Server  │         │   C++ Server    │
│   (Flask)       │         │   (httplib)     │
│   porta 5001    │         │   porta 5002    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │    Algoritmo CGNR         │
         │    (Reconstrução)         │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Datasets (H, G)                           │
│              30x30, 60x60 - Matrizes Sparse                  │
└─────────────────────────────────────────────────────────────┘
```

### Componentes

| Componente | Descrição |
|------------|-----------|
| `run_benchmark.py` | Orquestra servidores e clientes |
| `server/python_server.py` | Servidor Flask com solver CGNR em NumPy |
| `server/cpp_http_server.cpp` | Servidor HTTP C++ com solver CGNR em Eigen |
| `scripts/client_generator.py` | Gera jobs e envia requisições |
| `scripts/generate_report_html.py` | Gera relatório HTML científico |

---

## 🎬 Modo DEMO - Protocolo Científico

O modo `--demo` executa um protocolo de benchmark determinístico em 3 fases:

### ACT 1: Sanity Check (Warmup)
- **Objetivo**: Aquecer servidores e verificar funcionamento
- **Jobs**: 1 job por servidor
- **Dataset**: 30x30_g1
- **Ganho**: Fixo em 1.0

### ACT 2: The Race (Análise de Variabilidade)
- **Objetivo**: Medir desempenho e calcular desvio padrão
- **Jobs**: 3 repetições por dataset
- **Datasets**: Todos (30x30 e 60x60)
- **Ganho**: Fixo em 1.0
- **Concorrência**: 1 cliente (execução sequencial)

### ACT 3: Saturation (Teste de Stress)
- **Objetivo**: Testar comportamento sob carga
- **Jobs**: 5 repetições × 6 datasets × 3 clientes
- **Datasets**: Todos
- **Ganho**: Fixo em 1.0
- **Concorrência**: 3 clientes simultâneos
- **Monitoramento**: CPU e RAM do sistema

### Saída
Após execução, o relatório é gerado em:
```
execs/<timestamp>_DEMO/Relatorio_Cientifico.html
```

---

## 📁 Estrutura do Projeto

```
dis/
├── build/                    # Binários compilados
├── data/                     # Datasets (H, G matrices)
├── execs/                    # Resultados de experimentos
├── include/                  # Headers C++
├── scripts/
│   ├── client_generator.py   # Gerador de jobs
│   └── generate_report_html.py
├── server/
│   ├── cpp_http_server.cpp   # Servidor C++
│   └── python_server.py      # Servidor Python
├── src/                      # Código fonte C++
├── CMakeLists.txt
├── config.yaml               # Configuração do benchmark
├── requirements.txt
└── run_benchmark.py          # Script principal
```

---

## 📈 Métricas Coletadas

| Métrica | Descrição |
|---------|-----------|
| `solver_time_ms` | Tempo do algoritmo CGNR |
| `latency_ms` | Tempo total da requisição |
| `iterations` | Iterações até convergência |
| `final_error` | Erro residual final |
| `ram_peak_mb` | Uso de memória (Python) |
| `throughput` | Imagens/segundo |
| `speedup` | Razão Python/C++ |

---

## 🔍 Troubleshooting

### Erro: "CMake generator mismatch"
```powershell
Remove-Item -Recurse -Force build
# Recompile do zero
```

### Erro: "FileNotFoundError" ao iniciar benchmark
```powershell
# Verifique se o executável existe
Test-Path .\build\UltrasoundServerHTTP.exe
```

### Portas em uso
```powershell
# Verifique se portas 5001/5002 estão livres
netstat -ano | findstr "5001 5002"
```

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos e de pesquisa.
