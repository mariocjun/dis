# Projeto de Reconstrução de Ultrassom (DIS)

Este projeto implementa algoritmos de reconstrução de imagens de ultrassom (CGNR Standard e Pre-condicionado) em C++ com visualização em Python.

## Estrutura do Projeto

```
/projeto_dis/
|-- build_and_run.py       # Script principal de automação (Compila -> Roda -> Visualiza)
|-- config.yaml            # Configuração dos testes e parâmetros
|-- main.cpp               # Código fonte C++ principal
|-- solver_comparison.hpp  # Implementação dos solvers
|-- scripts/               # Scripts auxiliares de visualização
|-- data/                  # Dados de entrada (Matrizes H e vetores g)
|-- output_runs/           # [GERADO] Histórico de execuções
    |-- DD_MM_AAAA_.../
        |-- images/        # CSVs das imagens reconstruídas
        |-- metrics/       # CSVs de métricas e convergência
        |-- animations/    # GIFs gerados automaticamente
```

## Pré-requisitos

1.  **C++ Compiler**: Compatível com C++17 (MSVC, GCC, Clang).
2.  **CMake**: Para configuração do projeto.
3.  **Python 3.x**: Para automação e visualização.

### Dependências Python

O projeto utiliza um ambiente virtual (`.venv`) ou instalação global. As bibliotecas necessárias são:

```bash
pip install pyyaml imageio numpy matplotlib
```

## Como Rodar

A maneira recomendada é usar o script de automação, que gerencia todo o processo:

```bash
python build_and_run.py
```

### O que o script faz?

1.  **Lê Configurações**: Extrai parâmetros de `config.yaml` (tolerância, iterações, threads).
2.  **Compila**: Executa o CMake e builda o executável `UltrasoundBenchmark`.
3.  **Executa**: Roda o benchmark C++, salvando os resultados em uma pasta com timestamp:
    *   Exemplo: `output_runs/13_12_2025_04_10_tol_1e-4_ite_10_omp_8`
4.  **Visualiza**: Chama automaticamente o gerador de GIFs para criar animações das reconstruções nessa mesma pasta.

## Configuração (`config.yaml`)

Você pode ajustar os parâmetros da simulação no arquivo `config.yaml`:

*   `epsilon_tolerance`: Critério de parada do solver.
*   `max_iterations`: Número máximo de iterações.
*   `num_omp_threads`: Número de threads para paralelismo (OpenMP).
*   `datasets`: Define quais sinais/matrizes serão processados.

## Saída dos Dados

Cada execução cria uma nova pasta dentro de `output_runs/` para preservar o histórico.

*   **metrics/**: Contém CSVs com o histórico de convergência (erro residual, norma, etc.) e curva L.
*   **images/**: Contém os dados brutos das imagens reconstruídas (CSV) passo a passo (se ativado).
*   **animations/**: Contém os GIFs animados comparando a evolução da reconstrução.