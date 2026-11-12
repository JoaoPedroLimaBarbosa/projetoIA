# 🌍 Classificação de Países por Prioridade de Ajuda

Aplicação de Inteligência Artificial desenvolvida em Python para agrupar países de acordo com indicadores socioeconômicos, identificando diferentes perfis de desenvolvimento e possíveis prioridades de ajuda.

O projeto utiliza técnicas de aprendizado de máquina não supervisionado e disponibiliza uma interface interativa desenvolvida com Streamlit.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-KMeans-orange" />
  <img src="https://img.shields.io/badge/PCA-5%20Componentes-purple" />
  <img src="https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow" />
</p>

---

## 📌 Sobre o projeto

O objetivo é analisar indicadores socioeconômicos de diferentes países e agrupá-los conforme suas características.

A aplicação permite informar os indicadores de um país e obter uma classificação baseada no modelo treinado, além de comparar seus valores com as médias do grupo atribuído.

O projeto foi desenvolvido como atividade prática da disciplina de **Inteligência Artificial – Unidade III**, no curso de Engenharia de Software do UNIFSA.

## 🎯 Funcionalidades

- **Classificação de países:** identifica o grupo ao qual o país informado pertence.
- **Comparação de indicadores:** apresenta os valores informados e as médias dos países do grupo.
- **Visualização PCA:** apresenta os países em um gráfico com os componentes PC1 e PC2, destacando o novo país informado.
- **Sobre o modelo:** apresenta informações sobre a origem dos dados, técnicas utilizadas e configuração do modelo.
- **Interface interativa:** permite preencher os indicadores diretamente pelo navegador.

## 🧠 Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem principal |
| Pandas | Manipulação e análise dos dados |
| Scikit-learn | Construção do modelo de Machine Learning |
| StandardScaler | Padronização dos indicadores |
| PCA | Redução de dimensionalidade |
| KMeans | Agrupamento dos países |
| Joblib | Salvamento e carregamento do modelo |
| Matplotlib | Construção dos gráficos |
| Streamlit | Desenvolvimento da interface web |

## 📊 Base de dados

O projeto utiliza o conjunto de dados **Country-data**, disponibilizado publicamente no Kaggle.

- Quantidade de países: 167
- Quantidade de indicadores: 9
- Tipo de aprendizado: Não supervisionado
- Fonte: [Unsupervised Learning on Country Data – Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)

### Indicadores analisados

1. Mortalidade infantil (`child_mort`)
2. Exportações (`exports`)
3. Gastos com saúde (`health`)
4. Importações (`imports`)
5. Renda por pessoa (`income`)
6. Inflação (`inflation`)
7. Expectativa de vida (`life_expec`)
8. Filhos por mulher (`total_fer`)
9. PIB per capita (`gdpp`)

## ⚙️ Modelo de Machine Learning

O modelo utiliza um pipeline composto por três etapas:

1. **StandardScaler:** padroniza os indicadores para evitar que diferenças de escala prejudiquem a análise.
2. **PCA (Principal Component Analysis):** reduz a dimensionalidade dos dados, mantendo cinco componentes no pipeline.
3. **KMeans:** realiza o agrupamento dos países em três grupos.

### Grupos identificados

| Grupo | Perfil |
|---|---|
| Grupo 0 | Desenvolvidos |
| Grupo 1 | Em desenvolvimento |
| Grupo 2 | Prioridade de ajuda |

Os grupos representam perfis encontrados pelo algoritmo a partir dos indicadores analisados.

## 📁 Estrutura do projeto

```text
Projeto-IA/
│
├── README.md
│
└── projeto-ia/
    ├── app.py
    ├── requirements.txt
    ├── modelo_paises.pkl
    └── Country-data.csv
```

## 🚀 Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/JoaoPedroLimaBarbosa/Projeto-IA.git
```

### 2. Entre na pasta da aplicação

```bash
cd Projeto-IA/projeto-ia
```

### 3. Crie um ambiente virtual

```bash
python3 -m venv .venv
```

### 4. Ative o ambiente virtual

No macOS ou Linux:

```bash
source .venv/bin/activate
```

### 5. Instale as dependências

```bash
pip install -r requirements.txt
```

### 6. Execute a aplicação

```bash
streamlit run app.py
```

Após iniciar, acesse o endereço local apresentado pelo Streamlit, normalmente:

```text
http://localhost:8501
```

## ☁️ Publicação

A aplicação pode ser disponibilizada na internet utilizando o **Streamlit Community Cloud**, integrado ao GitHub.

Plataforma: [share.streamlit.io](https://share.streamlit.io/)

## ⚠️ Limitações e uso responsável

O modelo realiza agrupamentos com base nos indicadores disponíveis. Os resultados não representam uma avaliação definitiva da situação econômica ou social de um país.

A classificação deve ser interpretada como uma ferramenta de apoio à análise, considerando a qualidade dos dados, o contexto de cada país e a necessidade de avaliação humana.

---

## 👨‍💻 Autor

**João Pedro Lima Barbosa**

Estudante de Engenharia de Software – UNIFSA

Projeto acadêmico desenvolvido na disciplina de Inteligência Artificial.

---

<p align="center">
  Desenvolvido com Python, Machine Learning e Streamlit.
</p>
