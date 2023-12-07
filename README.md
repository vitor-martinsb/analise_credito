
## Projeto de Modelagem Preditiva

Este projeto consiste em uma implementação de modelagem preditiva usando Python para treinar e avaliar modelos de aprendizado de máquina. Os modelos são treinados para prever a variável alvo "TARGET" em um conjunto de dados fornecido.

### Conteúdo

1. [Estrutura do Projeto](#estrutura-do-projeto)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação](#instalação)
4. [Como Usar](#como-usar)
5. [Resultados](#resultados)
6. [Referências](#referências)

---

### Estrutura do Projeto

O projeto é organizado da seguinte maneira:

1) Carregue os arquivos localizados (https://www.kaggle.com/competitions/home-credit-default-risk) para a pasta data/input 
2) Crie o modelo executando o notebook model_ml.ipynb, ele irá gerar o modelo do random forest, caso opte por outro modelo, basta alterar o código
3) Execute o arquivo app.py e acesse http://127.0.0.1:8050/. O Dash será encontrado aqui:

![image](https://github.com/vitor-martinsb/analise_credito/assets/59899402/3959f1b8-261b-437b-bbfb-fb3a182d2377)

4) Carregue o arquivo exemplo_teste.csv ou application_test.csv no Dash e execute caso a colunas estejam corretas o dash será executado 

![image](https://github.com/vitor-martinsb/analise_credito/assets/59899402/c1d89b15-273f-4a1c-acfe-b8ead82499b0)

5) Realize a análise de crédito com base nos 3 scores e a probabilidade de compra

---

### Pré-requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

- pandas
- scikit-learn
- imbalanced-learn
- flask
- dash

