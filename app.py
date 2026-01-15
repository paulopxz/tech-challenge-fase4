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
    "Aplicação desenvolvida para o Tech Challenge – Fase 4. "
    "O modelo ARIMA é re-treinado dinamicamente para garantir "
    "compatibilidade no deploy em nuvem."
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
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    if "Último" not in df.columns:
        st.error("Coluna 'Último' não encontrada no CSV.")
        st.stop()

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
        "Quantidade insuficiente de dados para ajuste confiável do modelo ARIMA."
    )
else:
    try:
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

    except Exception as e:
        st.error("Erro ao ajustar ou prever com o modelo ARIMA.")
        st.exception(e)

# =========================
# RODAPÉ
# =========================
st.caption(
    "Modelo ARIMA definido e validado na Fase 2. "
    "Reajustado dinamicamente no app para garantir compatibilidade "
    "no ambiente Streamlit Cloud."
)
