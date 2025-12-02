# Projeto: Otimização de kNN com Redução de Protótipos (SRM-NN)
 ### Alunos 
 * **Juan Pabollo**
 * **Wellington Viana**
   
## 1. Visão Geral

Este projeto explora uma técnica de otimização para o classificador k-Nearest Neighbors (kNN) chamada **Redução de Protótipos**, baseada no princípio de **Minimização do Risco Estrutural (SRM)**.

O objetivo principal é reduzir drasticamente o tamanho do conjunto de treinamento, selecionando apenas os pontos mais informativos (protótipos), para então treinar um modelo kNN mais leve e rápido, mantendo uma alta acurácia de classificação.

## 2. Estrutura do Projeto

O código foi refatorado para uma melhor organização, dividindo as responsabilidades em módulos:

```
.
├── database/
│   └── banana.csv        # O conjunto de dados
├── utils/
│   ├── functions.py      # Contém a lógica da redução SRM-NN
│   └── plot.py           # Contém a função para visualização dos resultados
├── main.py               # Orquestrador principal do projeto
└── requirements.txt      # Dependências do projeto
```

-   **`main.py`**: O script principal que carrega os dados, aplica a redução, treina os modelos e chama a função de visualização.
-   **`utils/functions.py`**: Isola a função `srm_nn_reduce`, que é o núcleo da metodologia de redução de dados.
-   **`utils/plot.py`**: Contém a função `plot_informations` para gerar os gráficos comparativos.

## 3. A Metodologia: `srm_nn_reduce`

O coração do projeto é a função `srm_nn_reduce` (localizada em `utils/functions.py`). A sua lógica é a seguinte:

1.  **Identificar Pontos Críticos**: A função começa calculando a distância entre todos os pares de pontos que pertencem a classes opostas. A intuição é que os pontos mais próximos da fronteira de decisão são os mais importantes.
2.  **Priorizar os Menores Espaços**: Os pares com a menor distância entre si são processados primeiro, pois representam as áreas mais "confusas" ou críticas do espaço de features.
3.  **Construção Iterativa do Subconjunto**: Um subconjunto de treinamento (`J`) é iniciado vazio. Iterando sobre os pares (do mais próximo ao mais distante), seus pontos são adicionados ao subconjunto.
4.  **Critério de Parada Inteligente**: Após adicionar um novo par, o algoritmo testa se um classificador 1-NN treinado **apenas com o subconjunto `J`** já é capaz de classificar corretamente **todo o conjunto de treinamento original**.
5.  **Subconjunto Mínimo**: Assim que esse critério é atendido, o processo para. O resultado é um subconjunto de dados significativamente menor, mas que "representa" a estrutura do conjunto original.

## 4. Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a simulação.

### Passo 1: Configurar o Ambiente Virtual

Para manter as dependências isoladas, é altamente recomendado usar um ambiente virtual.

```bash
# Crie a pasta do ambiente virtual (ex: "venv")
python -m venv venv

# Ative o ambiente
# (No Windows)
.\venv\Scripts\activate
# (No macOS/Linux)
source venv/bin/activate
```

### Passo 2: Instalar as Dependências

Com o ambiente ativado, instale as bibliotecas listadas em `requirements.txt`.

```bash
# O pip usará o arquivo para instalar as versões corretas das bibliotecas
pip install -r requirements.txt
```

### Passo 3: Executar o Script Principal

Execute o arquivo `main.py`.

```bash
python main.py
```

### Passo 4: Interpretar os Resultados

O terminal exibirá as métricas de acurácia e redução. Uma janela se abrirá com os gráficos comparativos, permitindo uma análise visual da eficácia da técnica de redução.
