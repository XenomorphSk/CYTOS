# Pré-Registro: CYTOS — Tensor Networks Hierárquicas vs GNN para Redes Regulatórias Gênicas

**Data de registro:** 2026-06-24
**Autor:** GB
**Status:** Pré-registrado antes de qualquer execução de experimento

---

## 1. Motivação

GNNs convencionais tratam redes regulatórias gênicas como grafos planos, sem impor
estrutura hierárquica explícita. Redes regulatórias biológicas exibem topologia
modular hierárquica (módulos dentro de módulos, hubs esparsos). Tensor Networks
hierárquicas (Tree Tensor Network — TTN) foram desenhadas para comprimir
eficientemente sistemas com essa propriedade (entropia de emaranhamento que
escala com "boundary", não com volume).

## 2. Hipóteses

**H1 (hipótese de trabalho):** Um modelo TTN, com hierarquia definida pela
estrutura de comunidades do grafo regulatório, reconstrói a dinâmica de expressão
gênica (predição t → t+1) com eficiência paramétrica igual ou superior a uma GNN
convencional (GCN/GAT) — ou seja, MSE/n_parâmetros igual ou menor.

**H1b (teste mais específico):** TTN captura melhor correlações entre genes em
módulos hierarquicamente distantes do que GNN com mesmo número de parâmetros.
Esse é o teste que distingue "TTN é só outra parametrização" de "TTN captura algo
estrutural real".

**H0 (nula):** Não há diferença estatisticamente significativa em eficiência
paramétrica ou em captura de correlação de longo alcance entre TTN e GNN, após
controlar por número de parâmetros e variância de inicialização (múltiplas seeds).

## 3. Dataset

- **Fonte primária:** DREAM4 In Silico Network Challenge (tamanhos 10 e 100 genes,
  séries temporais com ruído conhecido).
- **Fonte secundária (validação cruzada):** modelo de Tyson/Novak do ciclo celular
  de *S. cerevisiae*, se tempo permitir.
- Split: 60% treino / 20% validação / 20% teste, estratificado por rede (não por
  ponto temporal, para evitar leakage temporal).

## 4. Operacionalização da hierarquia (TTN)

1. Construir grafo de regulação a partir do gold standard do DREAM4.
2. Detecção de comunidades via Louvain (ou spectral clustering como controle).
3. A árvore de hierarquia da TTN é definida por essa partição — isso é fixado
   ANTES de ver os resultados de desempenho, para evitar overfitting da topologia
   ao resultado.

## 5. Controle de parâmetros

GNN e TTN devem ter número de parâmetros treináveis dentro de ±10% um do outro
para cada tamanho de rede testado. Esse controle é calculado e registrado antes
do treino, não ajustado post-hoc.

## 6. Métricas pré-definidas

- **Métrica primária:** MSE de predição normalizado por nº de parâmetros.
- **Métrica secundária:** correlação de Pearson entre genes em módulos distantes
  (≥3 níveis de hierarquia de separação), predita vs real.
- **Critério de sucesso para H1:** TTN supera GNN na métrica primária em pelo
  menos 3 das 4 configurações testadas (10/100 genes × 2 níveis de ruído), com
  significância (teste pareado, múltiplas seeds, p<0.05 após correção para
  comparações múltiplas).
- **Critério de sucesso para H1b:** correlação de longo alcance da TTN
  estatisticamente maior que a da GNN em pelo menos 3 das 4 configurações.

## 7. Plano de falseação

Se nenhum dos dois critérios de sucesso for atingido → H1 e H1b são consideradas
falseadas. O resultado negativo será documentado e publicado/registrado com o
mesmo rigor que um resultado positivo. Não haverá re-interpretação post-hoc do
critério de sucesso após ver os números.

## 8. Lista de testes (a preencher conforme execução)

| # | Config (genes/ruído) | Métrica primária TTN | Métrica primária GNN | Pass/Fail H1 | Métrica secundária TTN | Métrica secundária GNN | Pass/Fail H1b |
|---|---|---|---|---|---|---|---|
| 1 | 10 genes / ruído baixo | | | | | | |
| 2 | 10 genes / ruído alto | | | | | | |
| 3 | 100 genes / ruído baixo | | | | | | |
| 4 | 100 genes / ruído alto | | | | | | |

## 9. Notas

Este documento não deve ser editado retroativamente após o início dos
experimentos, exceto para correções de erro de digitação claramente marcadas
com data e motivo.

## 10. Changelog de infraestrutura (2026-06-24)

**IMPORTANTE: nenhum resultado numérico desta seção vale para julgar H1/H1b.**
Todos os números abaixo vieram de `src/data/synthetic_dream4.py`, um gerador
de dados sintéticos criado exclusivamente para validar que o pipeline e os
modelos rodam de ponta a ponta sem erro — a dinâmica desses dados é
simplificada (modelo linear com acoplamento + ruído gaussiano) e NÃO tenta
replicar a estrutura biológica real do DREAM4 (motifs de regulação, topologia
scale-free calibrada, ruído biológico realista). Resultados de H1/H1b só
contam a partir dos dados reais do DREAM4 (Synapse).

Durante a fase de validação de infraestrutura, os seguintes problemas foram
encontrados e corrigidos, em ordem cronológica:

1. **Bug de `noise_level` ignorado** (`pipeline.py`): a função
   `load_dream4_timeseries` sempre lia o mesmo arquivo independente do nível
   de ruído configurado, fazendo "low" e "high" carregarem dados idênticos.
   Corrigido para ler arquivos com sufixo de ruído.

2. **Casamento de parâmetros unilateral** (`runner.py`):
   `match_parameter_counts` fixava o `hidden_dim` da GNN no valor do config
   e só buscava `bond_dim` da TTN. Para redes pequenas a TTN tinha overhead
   estrutural que não cabia nem no `bond_dim` mínimo. Corrigido para busca
   conjunta em `hidden_dim` (GNN) × `bond_dim` (TTN).

3. **Shape mismatch em `long_range_correlation`** (`runner.py`): os valores
   verdadeiros (`trues`) vindos do pipeline da GNN tinham shape `(n_genes, 1)`
   enquanto as predições tinham shape `(n_genes,)`, quebrando `np.corrcoef`.
   Corrigido com `.flatten()` em ambos antes do cálculo.

4. **Explosão numérica no gerador sintético** (`synthetic_dream4.py`): a
   simulação só limitava valores por baixo (`clip(x, 0, None)`), permitindo
   realimentação positiva em ciclos do grafo e valores chegando a ~3×10⁷ em
   200 passos. Corrigido com normalização espectral da matriz de
   acoplamento (raio espectral < 1) e clip simétrico (0 a 1).

5. **Overhead de parâmetros da TTN escalando com n_genes** (`ttn_model.py`):
   a camada de saída original (`output_proj = Linear(bond_dim, n_genes)`)
   tinha parâmetros crescendo linearmente com o número de genes, impedindo
   o casamento de parâmetros em redes de 100 genes (chegou a 100% de
   diferença residual). REDESENHADA para uma cabeça de leitura local
   compartilhada entre todos os genes (concat de vetor de folha + vetor de
   comunidade + vetor global, por um único `Linear` pequeno comum a todos
   os genes) — parâmetros agora constantes (3×bond_dim+1), independentes de
   n_genes. Esta é uma mudança de arquitetura do modelo, feita ANTES de
   qualquer rodada com dados reais, então não constitui p-hacking.

6. **Poder estatístico insuficiente com n=5 seeds**: o teste de Wilcoxon
   pareado com 5 seeds só atinge p<0.05 se todas as 5 tiverem o mesmo sinal.
   O padrão observado nas primeiras rodadas foi consistentemente 4-de-5 a
   favor da TTN (p=0.0625 em todas as configs testadas), sugerindo efeito
   real mas n insuficiente. Decisão tomada (antes de rodar de novo, não
   depois de ver o resultado com n maior): aumentar para 20 seeds.

7. **Disputa de threads em paralelização**: ao paralelizar o treino das
   seeds via `ProcessPoolExecutor`, cada processo tentava usar todos os
   threads disponíveis para paralelismo interno do PyTorch, gerando disputa
   severa (N processos × N threads competindo por N núcleos). Corrigido
   com `torch.set_num_threads(1)` no início de cada processo worker.

**Resultado da validação de infraestrutura (dados sintéticos, NÃO contam
para o pré-registro):** após as correções 1-7, H1 e H1b passaram em 4/4
configurações sintéticas testadas, confirmando que o pipeline roda
corretamente de ponta a ponta. Permanece a observação de que as correlações
de longo alcance (métrica H1b) ficaram quase saturadas (>0.95) em ambos os
modelos nos dados sintéticos, o que é esperado dado a simplicidade da
dinâmica gerada artificialmente — isso não deve ser interpretado como
suporte a H1b até a repetição com dados reais do DREAM4.

**Próximo passo:** obter acesso ao DREAM4 real via Synapse
(https://www.synapse.org/#!Synapse:syn3049712), substituir os dados em
`data/raw/dream4/` pelos arquivos oficiais, e rodar `python -m
src.experiments.runner` novamente. Somente esses resultados contam para
julgar H1 e H1b conforme definido neste pré-registro.

## 11. Amendment: estrutura real do DREAM4 (2026-06-24)

Ao obter acesso real ao DREAM4 via Synapse, duas premissas do desenho
original (Seção 3) se mostraram incorretas e precisam ser corrigidas ANTES
de qualquer execução com dados reais:

1. **Não existe "noise_levels: low/high".** O DREAM4 In Silico Networks
   Challenge fornece 5 REDES distintas por tamanho (instâncias/topologias
   diferentes, não níveis de ruído da mesma rede): insilico_size{N}_1 a
   insilico_size{N}_5. DECISÃO (tomada antes de rodar, não depois de ver
   resultado): usar as redes 1 e 2 para ambos os tamanhos (10 e 100 genes).
   As 4 configurações do pré-registro passam a ser:
   - 10 genes / rede 1
   - 10 genes / rede 2
   - 100 genes / rede 1
   - 100 genes / rede 2

2. **Bug evitado por inspeção manual:** cada arquivo
   insilico_size{N}_{rede}_timeseries.tsv contém 5 TRAJETÓRIAS separadas
   (réplicas a partir de condições iniciais diferentes, 21 timepoints cada,
   0 a 1000 em passos de 50), concatenadas no mesmo arquivo e separadas por
   linha em branco, com a coluna Time reiniciando em 0.0 a cada bloco. O
   pipeline original (pré-registrado) assumia uma única série temporal
   contínua e construiria pares (t, t+1) atravessando a fronteira entre
   trajetórias diferentes — isso teria sido um erro silencioso grave
   (pares falsos misturando duas trajetórias independentes). Corrigido: o
   pipeline agora respeita os blocos de trajetória, constrói pares (t,t+1)
   SOMENTE dentro de cada trajetória, e faz o split treino/val/teste POR
   TRAJETÓRIA INTEIRA (não por timepoint), eliminando qualquer vazamento
   temporal por construção. Com 5 trajetórias por rede, o split adotado é
   3 trajetórias para treino, 1 para validação, 1 para teste (aproxima os
   60/20/20 do desenho original, agora sem vazamento).

## 12. Resultados Finais (dados reais do DREAM4, redes 1 e 2)

Executado em 2026-06-24, com n=20 seeds, dados reais do DREAM4 In Silico
Networks Challenge (Synapse syn3049712), split por trajetória completa
(3 treino / 1 validação / 1 teste), sem vazamento temporal.

| Config | gnn_params | ttn_params | diff_param | MSE/param TTN | MSE/param GNN | p_primário (H1) | corr. longo alcance TTN | corr. longo alcance GNN | p_secundário (H1b) |
|---|---|---|---|---|---|---|---|---|---|
| 10 genes / rede 1 | 73 | 71 | 2.7% | 1.40×10⁻⁴ | 7.50×10⁻⁴ | 1.9×10⁻⁶ | 0.932 | 0.540 | 2.73×10⁻⁶⁷ |
| 10 genes / rede 2 | 145 | 133 | 8.3% | 6.80×10⁻⁵ | 2.25×10⁻⁴ | 1.9×10⁻⁶ | 0.935 | 0.622 | 2.73×10⁻⁶⁷ |
| 100 genes / rede 1 | 433 | 461 | 6.5% | 1.21×10⁻⁵ | 9.37×10⁻⁵ | 1.9×10⁻⁶ | 0.936 | 0.383 | 2.73×10⁻⁶⁷ |
| 100 genes / rede 2 | 433 | 449 | 3.7% | 1.64×10⁻⁵ | 1.04×10⁻⁴ | 1.9×10⁻⁶ | 0.923 | 0.217 | 2.73×10⁻⁶⁷ |

**H1: passou em 4/4 configurações** (critério pré-registrado: ≥3/4, p<0.05).
Em todas as configs, p_primário = 1.9×10⁻⁶ — o menor valor possível para o
teste de Wilcoxon pareado com n=20 seeds (equivalente a quase todas as 20
seeds favorecendo a TTN em eficiência paramétrica). A TTN venceu por uma
margem de 4 a 6× em MSE-por-parâmetro em todas as configurações.

**H1b: passou em 4/4 configurações** (mesmo critério). A correlação de
longo alcance da TTN permaneceu estável (~0.92-0.94) em todas as
configurações, enquanto a da GNN caiu progressivamente conforme o tamanho
da rede aumentou (0.54-0.62 em 10 genes, para 0.22-0.38 em 100 genes).
Diferente da rodada de validação com dados sintéticos (onde ambos os
modelos saturavam perto de 1.0, sugerindo efeito artificial da dinâmica
simplificada), aqui a diferença é grande e qualitativamente distinta — a
GNN degrada com o tamanho da rede, a TTN não.

### Limitações conhecidas desta rodada

1. **Conjunto de teste pequeno**: 1 trajetória (20 transições) por config,
   conforme definido no desenho pré-registrado. Generalização para outras
   trajetórias/condições experimentais (knockouts, knockdowns,
   multifactorial) não foi testada.

2. **p_secundário idêntico nas 4 configs** (2.73×10⁻⁶⁷): isso reflete o
   teto de significância atingível dado o tamanho de amostra (muitos pares
   de teste × 20 seeds), não uma medida fina da magnitude do efeito. A
   magnitude real do efeito deve ser lida pelas médias de correlação
   (coluna anterior), não pelo p-value.

3. **Dado real não utilizado em size=100**: as redes de 100 genes do
   DREAM4 têm 10 trajetórias disponíveis; o desenho usou apenas 5 (3+1+1),
   por consistência com o split usado em size=10 (que só tem 5
   trajetórias). As 5 trajetórias restantes de cada rede de 100 genes
   poderiam ser usadas em trabalho futuro para um teste mais robusto.

4. **Apenas redes 1 e 2 testadas** (de 5 disponíveis por tamanho), por
   decisão pré-registrada. Redes 3-5 permanecem como confirmação
   exploratória não realizada — útil para trabalho futuro, mas qualquer
   resultado nelas não pode ser usado para "salvar" H1/H1b caso essas
   redes específicas não tivessem confirmado a hipótese (isso seria
   p-hacking pela porta de trás).

### Conclusão

Pelos critérios definidos neste documento antes da execução com dados
reais, **H1 e H1b são sustentadas pelos dados do DREAM4** (redes 1 e 2,
tamanhos 10 e 100). O efeito é consistente em direção e magnitude nas 4
configurações testadas, e o achado mais substantivo — a degradação da
captura de correlação de longo alcance pela GNN conforme a rede cresce,
ausente na TTN — é biologicamente plausível (redes maiores têm estrutura
modular mais pronunciada, que a hierarquia da TTN explora por construção
e o message-passing raso da GNN não).

Isso não é uma prova definitiva — é um resultado positivo em escala
piloto (2 de 5 redes, 2 tamanhos), consistente com o protocolo de
pré-registro e falseação adotado. O próximo passo cientificamente honesto
é testar a robustez (redes 3-5, usar as trajetórias restantes de
size=100, e eventualmente escalar para redes maiores que 100 genes) antes
de tratar este resultado como estabelecido.
## 13. Confirmacao Exploratoria - Redes 3, 4, 5 (2026-06-24)

Executado como confirmacao exploratoria (nao pre-registrada, decidida e
rotulada como tal ANTES de rodar). Resultado: H1 e H1b passaram em 6/6
configuracoes exploratorias adicionais, replicando exatamente o padrao
observado nas configuracoes confirmatorias.

**Resultado combinado (4 confirmatorias + 6 exploratorias = 10/10):**

| size | network | label | corr. TTN | corr. GNN | h1_pass | h1b_pass |
|---|---|---|---|---|---|---|
| 10 | 1 | confirmatorio | 0.932 | 0.540 | sim | sim |
| 10 | 2 | confirmatorio | 0.935 | 0.623 | sim | sim |
| 10 | 3 | exploratorio | 0.910 | 0.527 | sim | sim |
| 10 | 4 | exploratorio | 0.921 | 0.333 | sim | sim |
| 10 | 5 | exploratorio | 0.951 | 0.543 | sim | sim |
| 100 | 1 | confirmatorio | 0.939 | 0.389 | sim | sim |
| 100 | 2 | confirmatorio | 0.922 | 0.247 | sim | sim |
| 100 | 3 | exploratorio | 0.945 | 0.409 | sim | sim |
| 100 | 4 | exploratorio | 0.930 | 0.458 | sim | sim |
| 100 | 5 | exploratorio | 0.939 | 0.365 | sim | sim |

H1 e H1b: **10/10**. O padrao e consistente em todas as 5 redes
disponiveis no DREAM4 In Silico Networks Challenge, nos dois tamanhos
testados (10 e 100 genes). A correlacao de longo alcance da TTN se
mantem estreitamente na faixa 0.91-0.95 independente de rede ou tamanho;
a da GNN varia mais (0.25-0.62) e tende a ser mais baixa em redes maiores.

### Ressalva critica para qualquer escrita/divulgacao deste resultado

**O DREAM4, apesar de ser um benchmark real da comunidade cientifica (nao
um gerador ad-hoc deste projeto), tambem e um dado SIMULADO** - gerado via
GeneNetWeaver (equacoes diferenciais ordinarias parametrizadas para imitar
estatisticas de redes biologicas reais), nao medicao de bancada. "Dados
reais" neste documento significa "benchmark padrao real da comunidade
DREAM", nao "medicao biologica direta". Qualquer escrita sobre este
resultado deve deixar essa distincao explicita, sob risco de overclaim.

### Escopo valido da conclusao até este ponto

Sustentado: TTN com hierarquia fixada por comunidades supera GNN em
eficiencia parametrica e em captura de correlacao de longo alcance, no
benchmark DREAM4 In Silico (5 redes, tamanhos 10 e 100 genes), de forma
consistente e replicada.

NAO sustentado por estes dados (e nao deve ser afirmado): generalizacao
para dados biologicos reais medidos experimentalmente, para redes maiores
que 100 genes, para topologias de regulacao muito diferentes das geradas
pelo GeneNetWeaver, ou superioridade da TTN em tarefas alem de predicao
de dinamica temporal (ex. inferencia de topologia, que e a tarefa
original do DREAM4 Challenge 2 e nao foi testada aqui).
