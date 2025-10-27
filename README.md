/projeto_dis/
|-- CMakeLists.txt
|-- main.cpp # Ponto de entrada principal (muito simples)
|-- config.yaml # Arquivo de configuração
|-- include/ # Pasta para arquivos de cabeçalho (.hpp)
| |-- types.hpp # Definições de structs (TestConfig, Results, Metrics)
| |-- config.hpp # Declaração da classe/struct de configuração
| |-- io_utils.hpp # Declarações das funções de I/O
| |-- solvers.hpp # Declarações dos algoritmos de reconstrução
| |-- utils.hpp # Declarações de funções utilitárias (normalize, etc.)
| |-- benchmark_runner.hpp # Declaração da função principal do benchmark
| |-- reporting.hpp # Declaração da função de geração de tabela
|-- src/ # Pasta para arquivos de implementação (.cpp)
| |-- config.cpp # Implementação do carregamento/parse do YAML
| |-- io_utils.cpp # Implementações das funções de I/O
| |-- solvers.cpp # Implementações dos algoritmos (CGNR, Precond, FISTA...)
| |-- utils.cpp # Implementações das funções utilitárias
| |-- benchmark_runner.cpp # Implementação da lógica principal do benchmark
| |-- reporting.cpp # Implementação da geração da tabela
|-- data/ # Seus dados CSV e BIN (como antes)
| |-- H-1.csv
| |-- G-1.csv
| |-- ...
|-- output/ # Nova pasta raiz para todos os resultados
| |-- logs/ # Arquivos de log (opcional)
| |-- results_csv/ # CSVs de convergência, L-curve, imagens
| |-- results_png/ # PNGs das imagens geradas pelo Python
| |-- summary_report.txt # Tabela final de resultados