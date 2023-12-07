
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

project-root/
│
├── data/
│ ├── input/
│ │ ├── application_train.csv
│ │ └── application_test.csv
│
│ ├── model_training.py
│ └── model_evaluation.py
│
├── model.py
├── model_ml.ipynb
├── explorer.ipynb
├── app.py
└── random_forest_model.pkl
│
└── README.md
└── .gitignore

- **data/:** Contém os arquivos de dados necessários para treinamento do modelo.
- **model/:** Contém os scripts Python para carregar dados, treinar modelos e avaliar a performance.
- **app/:** Contém a implementação da API Flask incorporada com Dash, incluindo o arquivo serializado do modelo Random Forest.

---

### Pré-requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

- pandas
- scikit-learn
- imbalanced-learn
- flask
- dash

