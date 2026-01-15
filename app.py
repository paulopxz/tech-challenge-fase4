import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - Ibovespa",
    layout="wide"
)

st.title("📈 Previsão do Ibovespa (ARIMA)")
st.write(
    "Aplicação desenvolvida para o Tech Challenge – Fase 4."
)

# =========================
# CAMINHO DOS DADOS
# =========================
DATA_PATH = Path("data/Dados Históricos - Ibovespa 2005-2025.csv")

# =========================
# CARREGAMENTO DOS DADOS
# =========================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="latin-1"
    )

    # Normalizar nomes
    df.columns = df.columns.str.strip()

    # Converter data
    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d.%m.%Y",
        errors="coerce"
    )

    # Converter preço
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


# =========================
# EXECUÇÃO
# =========================
df = carregar_dados()

st.write(f"📊 Total de registros carregados: {len(df)}")

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
st.subheader("🔮 Previsão do Próximo Log-Return")

if len(df_lr) < 50:
    st.warning(
        f"Quantidade insuficiente de dados para ARIMA. "
        f"Registros válidos: {len(df_lr)}"
    )
else:
    with st.spinner("Ajustando modelo ARIMA..."):
        modelo = ARIMA(
            df_lr["log_return"],
            order=(1, 0, 1)
        ).fit()

        previsao = modelo.forecast(steps=1)[0]

    st.metric(
        label="Log-return previsto",
        value=f"{previsao:.6f}"
    )

st.caption(
    "Modelo ARIMA definido na Fase 2 e ajustado dinamicamente "
    "no app para garantir compatibilidade em produção."
)
