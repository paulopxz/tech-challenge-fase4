import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

# =========================
# CONFIGURAÇÃO
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - Ibovespa",
    layout="wide"
)

st.title("📈 Previsão do Ibovespa")
st.write("Aplicação desenvolvida para o Tech Challenge – Fase 4")

DATA_PATH = Path("data/Dados Históricos - Ibovespa 2005-2025.csv")

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Converter data
    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Converter fechamento
    df["Fechamento"] = (
        df["Último"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df["Fechamento"] = pd.to_numeric(df["Fechamento"], errors="coerce")

    df = df.dropna(subset=["Data", "Fechamento"])
    df = df.sort_values("Data")

    return df

df = carregar_dados()

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
# PREVISÃO
# =========================
st.subheader("🔮 Previsão do Próximo Log-Return")

if len(df_lr) < 30:
    st.warning("Quantidade insuficiente de dados para ajuste confiável do modelo ARIMA.")
else:
    serie = df_lr["log_return"]

    modelo = ARIMA(serie, order=(1, 0, 1))
    modelo_ajustado = modelo.fit()

    previsao = modelo_ajustado.forecast(steps=1).iloc[0]

    st.metric(
        label="Log-return previsto",
        value=f"{previsao:.6f}"
    )

st.caption("Modelo ARIMA ajustado dinamicamente com dados históricos do Ibovespa.")
