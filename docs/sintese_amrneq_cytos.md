# Síntese Cruzada: Degenerescência de Proxies de Baixa Ordem em AMR-NEQ e CYTOS

**Autor:** GB
**Data:** Junho 2026
**Status:** Nota de síntese conceitual — conecta dois projetos independentes via um princípio estrutural comum, descoberto de forma independente em cada um.

---

## Resumo

AMR-NEQ (física quântica, caracterização de ruído NISQ) e CYTOS (tensor networks aplicadas a dinâmica de redes regulatórias gênicas) são projetos de domínios completamente diferentes. Mas, analisando os dois com o mesmo rigor de pré-registro e falseação, emergiu **o mesmo padrão estrutural matemático** em ambos, descoberto de forma totalmente independente:

> **Proxies de "entropia"/estrutura baseados em quantidades de baixa ordem (marginais pequenas, ou matrizes de peso locais não-canonicalizadas) degeneram — ficam estruturalmente cegos a diferenças reais — quando o sistema subjacente tem simetria, transitividade, ou amostra pouco diversa. A correção, nos dois casos, exige subir pra uma quantidade de ordem mais alta ou aumentar a diversidade da amostra testada.**

Isso **não é** uma afirmação de que biologia é literalmente um sistema quântico, ou que redes regulatórias gênicas obedecem holografia. É uma conexão estrutural — o mesmo tipo de patologia matemática (degenerescência de proxy de baixa ordem) aparecendo em dois formalismos que compartilham a mesma matemática de fundo (tensor networks, decomposição em subsistemas, entropia/informação).

---

## 1. O padrão no AMR-NEQ

**Teorema 3 (AMR-NEQ):** qualquer construção de proxy baseada em marginais de corpo fixo k é degenerada sempre que o estado possui uma simetria (permutação completa, equivalência por unitária local, ou transitividade de vértices) que atua transitivamente sobre subconjuntos de tamanho k. GHZ, W, estado de grafo estrela e anel falham todos pelo mesmo motivo estrutural — apesar de parecerem mecanismos diferentes na superfície.

**Consequência prática:** a correlação entre o proxy espectral `I_G` (baseado em marginais de 1 qubit) e a entropia de emaranhamento real `S_A` tem um teto estrutural (~0.3-0.6), confirmado em simulação **e em hardware real** (IBM `ibm_kingston`, 156 qubits), independente de topologia ou modelo de ruído testado.

**A mesma simetria que cega o proxy, em outro contexto, protege:** no Apêndice D do AMR-NEQ, a assimetria de fidelidade de Leung (sob amortecimento de amplitude) em redes de tensores perfeitos conectados é suprimida por ~100-1000x a cada tensor adicionado — não porque a distância de correção de erros cresce (hipótese testada e refutada por busca de força bruta), mas porque o **grupo de simetria que relaciona os setores lógicos cresce**, degenerando os operadores `W` e `W²` que sustentam o cancelamento de erro. **A mesma simetria estrutural que cega o proxy de baixa ordem (Parte I) é o que protege a fidelidade lógica (Parte II/Apêndice D)** — dois lados da mesma moeda matemática.

## 2. O mesmo padrão, descoberto de forma independente, no CYTOS

**Achado (Seção 17 do pré-registro CYTOS):** ao testar o piloto de entropia de bond (proxy simplificado, baseado em SVD de uma matriz de peso local não-canonicalizada) num único grafo pequeno isolado, a entropia degenerou entre comunidades — várias bateram no mesmo valor, quebrando a correlação de Spearman (entrada constante). Isso ocorreu especificamente quando a amostra era pequena, pouco diversa, e `bond_dim` era baixo (resolução limitada da métrica).

**A correção seguiu exatamente a mesma lógica do AMR-NEQ:**
- *Subir para quantidade de ordem mais alta:* implementamos canonicalização rigorosa (Seção 20) — em vez do proxy simplificado, uma decomposição QR genuína que produz entropia de von Neumann real (dentro de um escopo bem definido), análoga a abandonar marginais de corpo fixo pequeno em favor de uma quantidade que captura estrutura genuína.
- *Aumentar diversidade da amostra:* o resultado robusto de H2 (ρ=0.51, p=1.4×10⁻¹⁸) só apareceu ao agregar comunidades de **10 redes diferentes**, não de uma rede isolada — exatamente como o AMR-NEQ só conseguiu validar seu teto estrutural de forma robusta ao testar em **múltiplas condições independentes** (5 confirmações, incluindo 2 em hardware real).

**E há uma segunda conexão, estruturalmente mais profunda:** no CYTOS, H1/H1b mostram que a TTN vence quando recebe a hierarquia **correta** (comunidades reais) como estrutura — e H3 mostra que essa vantagem **desaparece inteiramente** (e até inverte) quando a hierarquia é **arbitrária**. Isso é o mesmo princípio do AMR-NEQ Apêndice D: **estrutura/simetria correta protege contra degenerescência; estrutura incorreta ou ausente não protege, e pode até prejudicar.**

## 3. O princípio estrutural unificador

| | AMR-NEQ | CYTOS |
|---|---|---|
| Proxy de baixa ordem que degenera | `I_G` (marginal de 1 qubit) | Entropia local de bond (proxy, sem canonicalização) |
| Causa da degenerescência | Simetria/transitividade do estado | Amostra pequena/pouco diversa + bond_dim baixo |
| Correção via ordem mais alta | (sugerido, não implementado) tensores de correlação multi-corpo | Canonicalização rigorosa via QR (implementada, Seção 20) |
| Correção via diversidade de amostra | 5 confirmações independentes, incl. hardware real | Agregação de 51 comunidades de 10 redes |
| Estrutura correta protege | Simetria crescente protege fidelidade lógica (redes de tensores perfeitos) | Hierarquia correta (comunidades reais) dá vantagem real (H1/H1b) |
| Estrutura incorreta não protege | (não testado diretamente neste eixo) | Hierarquia arbitrária remove/inverte a vantagem (H3, falseado) |

## 4. O que esta síntese NÃO afirma

- Não afirma que redes regulatórias gênicas são sistemas quânticos, nem que biologia obedece holografia ou correspondência de Ryu-Takayanagi.
- Não afirma que a entropia de bond do CYTOS é fisicamente equivalente à entropia de emaranhamento do AMR-NEQ — são formalismos matemáticos análogos (tensor networks, decomposição SVD, entropia de Shannon/von Neumann), não o mesmo fenômeno físico.
- A conexão é estrutural-matemática (o mesmo tipo de patologia e a mesma classe de correção aparecendo em dois domínios que compartilham ferramentas matemáticas), não uma afirmação de unificação física.

## 5. Por que isso importa

O valor desta síntese não é "provar uma teoria de tudo" — é demonstrar que a **disciplina metodológica** (pré-registro, falseação, ceticismo sobre correlações "boas demais", verificação automática de invariantes) aplicada consistentemente em dois domínios completamente diferentes revela **o mesmo tipo de armadilha matemática recorrente**. Isso é evidência de competência metodológica transferível entre domínios — exatamente o tipo de coisa que diferencia um pesquisador independente genuíno de alguém que só sabe aplicar uma ferramenta a um problema.

---

*Esta nota de síntese conecta `AMR-NEQ — Investigação em Andamento` / `AMR-NEQ v6.0` (física quântica) e `CYTOS` (tensor networks para biologia computacional), ambos documentos de pesquisa independente do mesmo autor, com pré-registro e código público.*
