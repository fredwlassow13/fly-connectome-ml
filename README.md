

Fly Connectome ML

> 🇧🇷 [Português](#português) | 🇺🇸 [English](#english)

## Português

### Sobre o Projeto

Pipeline para análise e aplicação de Machine Learning sobre dados de conectoma de Drosophila melanogaster.
Status: desenvolvimento — primeira etapa de preprocessing validada com um subconjunto de 10.000 neurônios.

Sobre o projeto
Este projeto tem como objetivo construir um pipeline reprodutível para preparar e analisar dados de conectividade neural em larga escala, criando uma representação adequada para experimentos posteriores de Machine Learning.
Como o conectoma completo possui um número elevado de neurônios e conexões, a primeira etapa experimental utiliza um subconjunto de 10.000 neurônios, selecionados de forma estratificada de acordo com a superclass de cada neurônio.
A estratégia permite testar e validar o pipeline em uma escala controlada antes de expandi-lo para conjuntos maiores.

Pipeline atual
O preprocessing desenvolvido até o momento segue:
Raw connectome
      │
      ▼
Filtragem dos nós
      │
      ▼
Cálculo de graus e contatos
      │
      ▼
166.700 nós retidos
      │
      ▼
Seleção estratificada
      │
      ▼
10.000 nós
      │
      ▼
Subgrafo induzido
      │
      ▼
90.859 arestas
      │
      ▼
Validação estrutural
      │
      ▼
Subgrafo validado para as próximas etapas

Dados utilizados
Os dados brutos utilizados pelo pipeline são organizados em:
data/raw/
├── annotations.feather
├── neurotransmitters.feather
└── edges.feather
annotations.feather
Contém as anotações dos neurônios, incluindo informações como:
* bodyId
* superclass
* type
* outras características disponíveis nas anotações.
neurotransmitters.feather
Contém as anotações relacionadas aos neurotransmissores dos neurônios.
edges.feather
Contém as conexões entre neurônios, incluindo:
* body_pre
* body_post
* weight
onde weight representa o número de contatos sinápticos associados à conexão.

Preprocessing
1. Filtragem e cálculo dos graus
A primeira etapa processa os nós retidos e percorre o arquivo de arestas para calcular:
* in_degree
* out_degree
* total_degree
* in_contacts
* out_contacts
O cálculo é realizado por streaming dos batches do arquivo Arrow, evitando carregar todas as arestas simultaneamente na memória.
Resultado
Retained nodes:        166,700
Retained edge rows:     25,582,938
Isolated nodes:                124
Density (directed):       9.206e-04
A distribuição de total_degree obtida foi:
count    166700
mean       1011.41
std        1648.82
min           0
25%         315
50%         574
75%        1027
max       70035

2. Seleção do subconjunto de 10.000 neurônios
Para os primeiros testes, foram selecionados 10.000 neurônios usando amostragem estratificada por superclass.
Parâmetros utilizados:
n = 10,000
strategy = stratified
seed = 42
A seleção preserva aproximadamente a composição das superclass do conjunto de 166.700 neurônios.
Principais classes na amostra:
Superclass	Nós
ol_intrinsic	5.363
cb_intrinsic	1.929
vnc_intrinsic	790
visual_projection	552
vnc_sensory	382
ol_sensory	366
cb_sensory	292
Total:
10,000 nós
A estratégia top_degree também foi implementada, mas não é utilizada como subconjunto principal, pois selecionaria preferencialmente hubs e introduziria um viés estrutural.

3. Construção do subgrafo induzido
Após a seleção dos 10.000 neurônios, o arquivo de arestas é percorrido novamente.
São mantidas somente as conexões em que:
body_pre ∈ subset
AND
body_post ∈ subset
O resultado é um subgrafo induzido pelos 10.000 neurônios selecionados.
Resultado
Retained nodes:      10,000
Retained edge rows:   90,859
Isolated nodes:          311
Density (directed):  9.087e-04
A densidade do subgrafo permanece próxima à observada no conjunto maior:
166.700 nós → 9.206e-04
 10.000 nós → 9.087e-04
Isso indica que a amostragem estratificada não produziu uma alteração drástica na densidade estrutural da rede.

Validação
O subgrafo é validado automaticamente antes de ser utilizado nas próximas etapas.
As verificações incluem:
Integridade dos nós
* bodyId únicos;
* exatamente 10.000 nós.
Integridade das arestas
* todo body_pre pertence ao subconjunto;
* todo body_post pertence ao subconjunto.
Graus
Os seguintes valores são recalculados diretamente a partir das arestas e comparados com os valores armazenados:
in_degree
out_degree
in_contacts
out_contacts
total_degree
Consistência global
Também são verificadas:
sum(out_degree) == número de arestas
sum(in_degree)  == número de arestas

sum(out_contacts) == sum(weight)
sum(in_contacts)  == sum(weight)
Resultado da validação atual:
=== Subgraph validation ===
Nodes: 10,000
Edges: 90,859
Edge endpoints:      OK
in_degree:           OK
out_degree:          OK
in_contacts:         OK
out_contacts:        OK
total_degree:        OK
Isolated nodes:      311
Global edge count:   OK
Global contact sum:  OK

[OK] Subgraph validation passed.

Arquivos gerados
Após o preprocessing, os principais arquivos são:
data/processed/
├── edges_subset_10000_stratified.parquet
├── nodes_subset_10000_stratified.parquet
├── nodes_subset_10000_stratified_with_degrees.parquet
└── nodes_with_degrees.parquet
nodes_with_degrees.parquet
Nós do conjunto retido de 166.700 neurônios com suas métricas de conectividade calculadas no grafo maior.
nodes_subset_10000_stratified.parquet
Os 10.000 neurônios selecionados pela estratégia estratificada.
nodes_subset_10000_stratified_with_degrees.parquet
Os mesmos 10.000 neurônios, porém com os graus e contatos recalculados exclusivamente dentro do subgrafo de 90.859 arestas.
edges_subset_10000_stratified.parquet
Arestas internas ao subconjunto de 10.000 neurônios.

Estrutura atual do projeto
```text
fly-connectome-ml/
│
├── connectome/
│   └── preprocessing.py
│
├── data/
│   ├── raw/
│   │   ├── annotations.feather
│   │   ├── neurotransmitters.feather
│   │   └── edges.feather
│   │
│   └── processed/
│       ├── edges_subset_10000_stratified.parquet
│       ├── nodes_subset_10000_stratified.parquet
│       ├── nodes_subset_10000_stratified_with_degrees.parquet
│       └── nodes_with_degrees.parquet
│
└── README.md
```text

Os arquivos de dados brutos e processados podem ser excluídos do versionamento caso sejam grandes ou possuam restrições de redistribuição. Nesse caso, o repositório deve documentar sua origem e o procedimento necessário para reproduzi-los.

Reprodutibilidade
A seleção dos 10.000 neurônios utiliza uma seed fixa:
seed = 42
Portanto, a seleção é determinística enquanto os dados de entrada e a implementação permanecerem iguais.
O pipeline também utiliza processamento por batches para percorrer o arquivo de arestas.

Estado atual
Concluído
*  Carregamento dos dados de annotations
*  Carregamento dos dados de neurotransmissores
*  Processamento das arestas
*  Cálculo de graus do conjunto retido
*  Validação da conectividade dos nós
*  Seleção estratificada de 10.000 neurônios
*  Construção do subgrafo induzido
*  Recalculação dos graus dentro do subgrafo
*  Validação estrutural do subgrafo
*  Persistência dos nós e arestas processados
Próximas etapas
*  Integração das anotações de neurotransmissores
*  Análise dos neurônios sem anotação de neurotransmissor
*  Integração das demais características neuronais
*  Definição das features para Machine Learning
*  Construção da matriz final de features
*  Definição dos problemas de classificação/regressão
*  Treinamento e avaliação dos modelos
*  Comparação de diferentes estratégias de representação do conectoma
*  Avaliação em subconjuntos maiores do conectoma

Objetivo da etapa de 10.000 neurônios
Os 10.000 neurônios não representam necessariamente a configuração final do experimento.
Este subconjunto funciona como uma escala experimental inicial, permitindo validar:
1. o pipeline de preprocessing;
2. a integridade do grafo;
3. a integração das diferentes fontes de anotação;
4. a construção das features;
5. o funcionamento dos modelos de Machine Learning.
Após a validação dessas etapas, o pipeline poderá ser executado em subconjuntos maiores ou, quando viável, sobre uma parcela maior do conectoma.


## English

### About

Fly Connectome ML
Pipeline for analyzing and applying Machine Learning to Drosophila melanogaster connectome data.
Status: in development — first preprocessing stage validated using a 10,000-neuron subset.

About the project
This project aims to build a reproducible pipeline for preparing and analyzing large-scale neural connectivity data, creating a suitable representation for subsequent Machine Learning experiments.
Because the complete connectome contains a large number of neurons and connections, the first experimental stage uses a subset of 10,000 neurons, selected using stratified sampling according to each neuron's superclass.
This strategy allows the pipeline to be tested and validated at a controlled scale before being expanded to larger datasets.

Current pipeline
The preprocessing pipeline developed so far follows:
Raw connectome
      │
      ▼
Node filtering
      │
      ▼
Degree and contact computation
      │
      ▼
166,700 retained nodes
      │
      ▼
Stratified sampling
      │
      ▼
10,000 nodes
      │
      ▼
Induced subgraph
      │
      ▼
90,859 edges
      │
      ▼
Structural validation
      │
      ▼
Validated subgraph for subsequent stages

Data
The raw data used by the pipeline are organized as:
data/raw/
├── annotations.feather
├── neurotransmitters.feather
└── edges.feather
annotations.feather
Contains neuron annotations, including information such as:
* bodyId
* superclass
* type
* other available neuron characteristics.
neurotransmitters.feather
Contains neurotransmitter-related annotations for the neurons.
edges.feather
Contains connections between neurons, including:
* body_pre
* body_post
* weight
where weight represents the number of synaptic contacts associated with a connection.

Preprocessing
1. Node filtering and degree computation
The first stage processes the retained nodes and streams through the edge file to compute:
* in_degree
* out_degree
* total_degree
* in_contacts
* out_contacts
The edge data are processed batch by batch from the Arrow file, avoiding loading all edges into memory simultaneously.
Result
Retained nodes:        166,700
Retained edge rows:     25,582,938
Isolated nodes:                124
Density (directed):       9.206e-04
The resulting total_degree distribution was:
count    166700
mean       1011.41
std        1648.82
min           0
25%         315
50%         574
75%        1027
max       70035

2. Selection of the 10,000-neuron subset
For the first experiments, 10,000 neurons were selected using stratified sampling by superclass.
Parameters:
n = 10,000
strategy = stratified
seed = 42
The selection approximately preserves the superclass composition of the 166,700-neuron retained dataset.
Main classes in the sample:
Superclass	Neurons
ol_intrinsic	5,363
cb_intrinsic	1,929
vnc_intrinsic	790
visual_projection	552
vnc_sensory	382
ol_sensory	366
cb_sensory	292
Total:
10,000 nodes
A top_degree strategy has also been implemented, but it is not used as the main subset because it preferentially selects highly connected hubs and would introduce a strong structural bias.

3. Induced subgraph construction
After selecting the 10,000 neurons, the edge file is traversed again.
Only connections satisfying:
body_pre ∈ subset
AND
body_post ∈ subset
are retained.
The result is an induced subgraph containing the selected 10,000 neurons.
Result
Retained nodes:      10,000
Retained edge rows:   90,859
Isolated nodes:          311
Density (directed):  9.087e-04
The subgraph density remains close to the density observed in the larger retained graph:
166,700 nodes → 9.206e-04
 10,000 nodes → 9.087e-04
This indicates that the stratified sampling did not produce a drastic change in the network's structural density.

Validation
The induced subgraph is automatically validated before being used in subsequent stages.
The validation includes:
Node integrity
* unique bodyId values;
* exactly 10,000 nodes.
Edge integrity
* every body_pre belongs to the selected subset;
* every body_post belongs to the selected subset.
Degree consistency
The following values are recalculated directly from the edges and compared against the stored node-level values:
in_degree
out_degree
in_contacts
out_contacts
total_degree
Global consistency
The following relationships are also verified:
sum(out_degree) == number of edges
sum(in_degree)  == number of edges

sum(out_contacts) == sum(weight)
sum(in_contacts)  == sum(weight)
Current validation result:
=== Subgraph validation ===
Nodes: 10,000
Edges: 90,859
Edge endpoints:      OK
in_degree:           OK
out_degree:          OK
in_contacts:         OK
out_contacts:        OK
total_degree:        OK
Isolated nodes:      311
Global edge count:   OK
Global contact sum:  OK

[OK] Subgraph validation passed.

Generated files
After preprocessing, the main generated files are:
data/processed/
├── edges_subset_10000_stratified.parquet
├── nodes_subset_10000_stratified.parquet
├── nodes_subset_10000_stratified_with_degrees.parquet
└── nodes_with_degrees.parquet
nodes_with_degrees.parquet
Nodes from the 166,700-neuron retained dataset with connectivity metrics computed over the larger graph.
nodes_subset_10000_stratified.parquet
The 10,000 neurons selected using the stratified sampling strategy.
nodes_subset_10000_stratified_with_degrees.parquet
The same 10,000 neurons, with degrees and synaptic contacts recalculated exclusively within the 90,859-edge induced subgraph.
edges_subset_10000_stratified.parquet
Edges connecting neurons within the selected 10,000-neuron subset.

Current project structure

```text
fly-connectome-ml/
│
├── connectome/
│   └── preprocessing.py
│
├── data/
│   ├── raw/
│   │   ├── annotations.feather
│   │   ├── neurotransmitters.feather
│   │   └── edges.feather
│   │
│   └── processed/
│       ├── edges_subset_10000_stratified.parquet
│       ├── nodes_subset_10000_stratified.parquet
│       ├── nodes_subset_10000_stratified_with_degrees.parquet
│       └── nodes_with_degrees.parquet
│
└── README.md
```text

The raw and processed data files may be excluded from version control if they are large or subject to redistribution restrictions. In that case, the repository should document their source and the procedure required to reproduce them.

Reproducibility
The selection of the 10,000 neurons uses a fixed random seed:
seed = 42
Therefore, the selection is deterministic as long as the input data and implementation remain unchanged.
The pipeline also processes the edge data batch by batch.

Current status
Completed
*  Load neuron annotations
*  Load neurotransmitter annotations
*  Process edge data
*  Compute degrees for the retained graph
*  Validate neuron connectivity
*  Select 10,000 neurons using stratified sampling
*  Build the induced subgraph
*  Recalculate degrees within the subgraph
*  Validate the subgraph structure
*  Persist processed nodes and edges
Next steps
*  Integrate neurotransmitter annotations
*  Analyze neurons without neurotransmitter annotations
*  Integrate additional neuronal characteristics
*  Define Machine Learning features
*  Build the final feature matrix
*  Define classification/regression tasks
*  Train and evaluate Machine Learning models
*  Compare different connectome representation strategies
*  Evaluate the pipeline on larger subsets of the connectome

Purpose of the 10,000-neuron stage
The 10,000-neuron subset is not necessarily the final experimental configuration.
This subset serves as an initial experimental scale, allowing us to validate:
1. the preprocessing pipeline;
2. graph integrity;
3. integration of different annotation sources;
4. feature construction;
5. Machine Learning model execution.
After these stages have been validated, the pipeline can be applied to larger subsets or, when feasible, to a larger portion of the connectome.
