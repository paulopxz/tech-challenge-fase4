import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - Ibovespa",
    layout="wide"
)

st.title("📈 Previsão do Ibovespa (ARIMA)")
st.write(
    "Aplicação desenvolvida para o Tech Challenge – Fase 4. "
    "Modelo ARIMA treinado sobre log-retornos do Ibovespa."
)

# =========================
# CAMINHOS
# =========================
DATA_PATH = Path("data/Dados Históricos - Ibovespa 2005-2025.csv")
MODEL_PATH = Path("model/modelo_ibov.pkl")

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)

    # Normalizar nomes das colunas
    df.columns = df.columns.str.strip()

    # Converter coluna de data (formato brasileiro)
    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Verificação da coluna esperada
    if "Último" not in df.columns:
        st.error("Coluna 'Último' não encontrada no arquivo CSV.")
        st.stop()

    # Converter coluna de preço para float
    df["Fechamento"] = (
        df["Último"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df["Fechamento"] = pd.to_numeric(df["Fechamento"], errors="coerce")

    # Limpeza final
    df = df.dropna(subset=["Data", "Fechamento"])
    df = df.sort_values("Data")

    return df


# =========================
# CARREGAMENTO DO MODELO
# =========================
@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


# =========================
# EXECUÇÃO
# =========================
df = carregar_dados()
modelo = carregar_modelo()

# =========================
# FEATURE ENGINEERING
# =========================
df["log_return"] = np.log(df["Fechamento"]).diff()
df_lr = df.dropna(subset=["log_return"])

# =========================
# VISUALIZAÇÃO
# =========================
st.subheader("📊 Série Histórica do Ibovespa")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Data"], df["Fechamento"])
ax.set_xlabel("Data")
ax.set_ylabel("Ibovespa")
ax.grid(True)

st.pyplot(fig)

# =========================
# PREVISÃO COM ARIMA
# =========================
st.subheader("🔮 Previsão do Próximo Log-Return (ARIMA)")

if len(df_lr) < 30:
    st.warning(
        "Série histórica insuficiente para gerar previsão confiável."
    )
else:
    try:
        previsao = modelo.forecast(steps=1)[0]

        st.metric(
            label="Log-return previsto",
            value=f"{previsao:.6f}"
        )

    except Exception as e:
        st.error("Erro ao gerar previsão com o modelo ARIMA.")
        st.exception(e)

# =========================
# RODAPÉ
# =========================
st.caption(
    "Modelo ARIMA treinado na Fase 2 do Tech Challenge, "
    "aplicado em ambiente Streamlit Cloud na Fase 4."
)


