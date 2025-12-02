# kNN com Redução de Dados SRM (Structural Risk Minimization)
### Alunos

* **Juan Pabollo**
* **Wellington Viana**

Este projeto demonstra a implementação do algoritmo k-Nearest Neighbors (kNN) com uma técnica de redução de dados baseada no princípio de Minimização do Risco Estrutural (SRM). O objetivo é reduzir o tamanho do conjunto de treinamento, removendo pontos redundantes, e ainda assim manter (ou até melhorar) a acurácia do classificador.

## Como Funciona

O código realiza os seguintes passos:

1.  **Carrega o Dataset**: Utiliza o arquivo `banana.csv`, que contém pontos de dados com coordenadas `x`, `y` e uma `class`.
2.  **Divide os Dados**: Separa o conjunto de dados em 70% para treinamento e 30% para teste.
3.  **Aplica a Redução SRM**: Uma função `srm_nn_reduce` é aplicada ao conjunto de treinamento. Essa função busca um subconjunto de dados que seja suficiente para classificar corretamente todos os outros pontos do conjunto de treinamento original. A ideia é manter apenas os pontos mais "informativos".
4.  **Treina Dois Modelos kNN**:
    *   Um modelo é treinado com o conjunto de dados de treinamento **original** e completo.
    *   Outro modelo é treinado com o conjunto de dados de treinamento **reduzido** pela função SRM.
5.  **Avalia e Compara**:
    *   Ambos os modelos são avaliados no mesmo conjunto de teste.
    *   A acurácia de cada modelo é calculada e impressa.
    *   O percentual de redução do conjunto de dados é exibido.
6.  **Visualiza os Resultados**: Dois gráficos são plotados para mostrar a distribuição dos pontos de dados: um para o conjunto de treinamento original e outro para o conjunto reduzido, permitindo uma comparação visual da "limpeza" dos dados.

## Como Executar o Projeto

### 1. Pré-requisitos

-   Python 3.x instalado.

### 2. Crie um Ambiente Virtual

É uma boa prática criar um ambiente virtual para isolar as dependências do projeto.

```bash
# Crie um ambiente virtual na pasta do projeto
python -m venv venv

# Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

### 3. Instale as Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias a partir do arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Execute o Código

Para rodar o script principal, execute o seguinte comando no seu terminal:

```bash
python main.py
```

### 5. Resultados Esperados

Após a execução, você verá no terminal:

-   O tamanho do conjunto de treinamento original.
-   O tamanho do conjunto de treinamento reduzido e o percentual de redução.
-   A acurácia do modelo kNN com o conjunto original.
-   A acurácia do modelo kNN com o conjunto reduzido.

Além disso, uma janela com dois gráficos será exibida, mostrando a comparação visual dos conjuntos de dados.
