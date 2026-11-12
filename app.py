"""Aplicativo de classificação de países por prioridade de ajuda.

Inteligência Artificial · Unidade III · UNIFSA · Engenharia de Software · 2026.2
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# 1. Configuração da página
st.set_page_config(
    page_title="Classificação de países",
    page_icon="🌍",
    layout="wide"
)

# Caminhos dos arquivos: todos devem ficar na mesma pasta deste app.py
PASTA_APP = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA_APP / "modelo_paises.pkl"
CAMINHO_DADOS = PASTA_APP / "Country-data.csv"


# 2. Carregamento dos arquivos
@st.cache_resource
def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        st.error("Não foi possível encontrar o arquivo modelo_paises.pkl.")
        st.caption(f"Caminho procurado: {CAMINHO_MODELO}")
        st.stop()

    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def carregar_dados():
    if not CAMINHO_DADOS.exists():
        st.error("Não foi possível encontrar o arquivo Country-data.csv.")
        st.caption(f"Caminho procurado: {CAMINHO_DADOS}")
        st.info(
            "Envie o arquivo Country-data.csv para o GitHub, "
            "na mesma pasta do app.py e do modelo_paises.pkl."
        )
        st.stop()

    return pd.read_csv(CAMINHO_DADOS)


pacote = carregar_modelo()
modelo = pacote["modelo"]       # StandardScaler -> PCA -> KMeans
colunas = pacote["colunas"]     # ordem usada no treinamento
nomes = pacote["nomes"]         # nome de cada grupo

dados = carregar_dados()

# Confere se a base contém todas as variáveis utilizadas pelo modelo.
colunas_ausentes = [coluna for coluna in colunas if coluna not in dados.columns]
if colunas_ausentes:
    st.error(f"Colunas ausentes no Country-data.csv: {colunas_ausentes}")
    st.stop()

X = dados[colunas].copy()

# O próprio pipeline calcula os grupos e as coordenadas PCA.
grupos_base = modelo.predict(X)
coordenadas_base = modelo[:-1].transform(X)

# 3. Descrição dos campos e valores iniciais
CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB per capita)", 35.0),
    "health": ("Gasto com saúde (% do PIB per capita)", 6.3),
    "imports": ("Importações (% do PIB per capita)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}

ROTULOS = {
    "child_mort": "Mortalidade infantil",
    "exports": "Exportações",
    "health": "Gasto com saúde",
    "imports": "Importações",
    "income": "Renda por pessoa",
    "inflation": "Inflação",
    "life_expec": "Expectativa de vida",
    "total_fer": "Filhos por mulher",
    "gdpp": "PIB per capita",
}


# 4. Interface
st.title("🌍 Classificação de países")
st.write(
    "Informe os indicadores socioeconômicos de um país para estimar "
    "seu grupo e comparar seu perfil com a média dos países desse grupo."
)

nome_pais = st.text_input("Nome do país", value="País informado")

valores = {}
esquerda, direita = st.columns(2)

for i, coluna in enumerate(colunas):
    rotulo, inicial = CAMPOS[coluna]
    with esquerda if i % 2 == 0 else direita:
        valores[coluna] = st.number_input(
            rotulo,
            value=float(inicial),
            key=coluna
        )


# 5. Classificação, médias do grupo e gráfico PCA
if st.button("Classificar", type="primary"):
    entrada = pd.DataFrame([valores])[colunas]

    # Classifica o país informado usando o pipeline completo.
    grupo = int(modelo.predict(entrada)[0])
    nome_grupo = nomes.get(grupo, f"Grupo {grupo}")

    st.success(f"{nome_pais or 'País informado'} — Grupo {grupo}: {nome_grupo}")

    # Atividade complementar 1: médias dos indicadores do grupo.
    dados_com_grupo = dados.copy()
    dados_com_grupo["grupo"] = grupos_base

    medias_grupo = (
        dados_com_grupo.loc[dados_com_grupo["grupo"] == grupo, colunas]
        .mean()
    )

    comparacao = pd.DataFrame({
        "Indicador": [ROTULOS.get(c, c) for c in colunas],
        "País informado": [valores[c] for c in colunas],
        "Média do grupo": [medias_grupo[c] for c in colunas],
    })

    st.subheader("Comparação com a média do grupo")
    st.write(
        f"Perfil médio calculado a partir de "
        f"{int((grupos_base == grupo).sum())} países do grupo **{nome_grupo}**."
    )
    st.dataframe(
        comparacao.style.format({
            "País informado": "{:,.2f}",
            "Média do grupo": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    # Atividade complementar 2: posiciona o país no plano PC1 x PC2.
    # modelo[:-1] aplica StandardScaler e PCA, sem executar o KMeans.
    coordenadas_pais = modelo[:-1].transform(entrada)
    pc1_pais = coordenadas_pais[0, 0]
    pc2_pais = coordenadas_pais[0, 1]

    st.subheader("País no plano PC1 × PC2")
    st.caption(
        "O gráfico utiliza somente os dois primeiros componentes para "
        "visualização. O agrupamento continua utilizando os cinco componentes "
        "retidos pelo PCA no pipeline."
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for grupo_plot in sorted(set(grupos_base)):
        indices = [i for i, g in enumerate(grupos_base) if g == grupo_plot]
        ax.scatter(
            coordenadas_base[indices, 0],
            coordenadas_base[indices, 1],
            alpha=0.65,
            s=45,
            label=f"Grupo {grupo_plot}: {nomes.get(int(grupo_plot), grupo_plot)}"
        )

    ax.scatter(
        pc1_pais,
        pc2_pais,
        color="red",
        marker="*",
        s=260,
        edgecolors="black",
        linewidths=0.8,
        label=f"{nome_pais or 'País informado'} (novo)",
        zorder=5
    )
    ax.annotate(
        nome_pais or "País informado",
        (pc1_pais, pc2_pais),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Países agrupados e país informado")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# 6. Sobre o modelo — documentação inicial (model card)
with st.expander("Sobre o modelo", expanded=False):
    st.markdown(
        """
        **Objetivo:** agrupar países por indicadores socioeconômicos para
        apoiar a análise de prioridades de ajuda.

        **Origem dos dados:** conjunto *Country-data*, disponibilizado no
        Kaggle: [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data).

        **Base utilizada:** 167 países e 9 indicadores socioeconômicos.

        **Data de treinamento:** não registrada no arquivo do modelo fornecido.

        **Pré-processamento e técnicas:** StandardScaler para padronização,
        PCA para redução de dimensionalidade e KMeans para agrupamento.

        **Componentes do PCA:** 5 componentes retidos pelo pipeline
        (configurado com `n_components=0.9`, preservando 90% da variância).

        **Número de grupos:** 3.

        **Nomes dos grupos:** Desenvolvidos, Em desenvolvimento e
        Prioridade de ajuda.

        **Limitações e uso responsável:** o agrupamento é uma ferramenta de
        apoio à análise, não uma decisão definitiva sobre a distribuição de
        recursos. Os resultados dependem da qualidade, da atualidade e da
        procedência dos dados, além dos indicadores escolhidos. Recomenda-se
        análise humana e consideração do contexto de cada país.
        """
    )
