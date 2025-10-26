1. O que estamos reconstruindo? A onda ou a imagem?
   Estamos reconstruindo a imagem.

Pense no processo desta forma:

* O Sinal de Entrada (g): Este é o "eco" bruto, a "onda" que o transdutor capta. É um vetor de dados que representa a variação de pressão ao longo do tempo para cada elemento do transdutor. É a informação bruta e "suja".


* O Processo de Reconstrução (Solver): O algoritmo CGNR é o cérebro da operação. Ele pega esse sinal bruto (g) e, usando o modelo do sistema (H), trabalha "de trás para frente" para descobrir quais pontos no espaço de interesse (os pixels da imagem) poderiam ter gerado aquele eco específico. O resultado desse processo é o vetor f.


* O Resultado (f): Este vetor f é a imagem reconstruída. Cada elemento de f corresponde à intensidade de um pixel na imagem final. Para uma imagem 60x60, f terá 3600 valores.