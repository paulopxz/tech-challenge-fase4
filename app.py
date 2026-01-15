import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURAÇÃO
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - Ibovespa",
    layout="wide"
)

st.title("📈 Previsão do Ibovespa")
st.write("Modelo treinado na Fase 2 e aplicado em Streamlit")

# =========================
# CAMINHOS
# =========================
DATA_PATH = Path("data/Dados Históricos - Ibovespa 2005-2025.csv")
MODEL_PATH = Path("model/modelo_ibov.pkl")

# =========================
# CARREGAR DADOS (REAL)
# =========================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(
        DATA_PATH,
        sep=",",
        encoding="latin-1"
    )

    # Corrigir nomes quebrados
    df = df.rename(columns={
        'ï»¿"Data"': 'Data',
        'Ã\x9altimo': 'Ultimo',
        'Ãltimo': 'Ultimo'
    })

    # Converter data
    df["Data"] = pd.to_datetime(
        df["Data"],
        format="%d.%m.%Y",
        errors="coerce"
    )

    # Garantir tipo numérico
    df["Ultimo"] = pd.to_numeric(df["Ultimo"], errors="coerce")

    df = df.dropna(subset=["Data", "Ultimo"])
    df = df.sort_values("Data").reset_index(drop=True)

    return df

# =========================
# CARREGAR MODELO
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
df["log_return"] = np.log(df["Ultimo"]).diff()
df_lr = df.dropna(subset=["log_return"])

# =========================
# GRÁFICO
# =========================
st.subheader("📊 Série Histórica do Ibovespa")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Data"], df["Ultimo"])
ax.set_xlabel("Data")
ax.set_ylabel("Ibovespa")
ax.grid(True)

st.pyplot(fig)

# =========================
# PREVISÃO
# =========================
st.subheader("🔮 Previsão do próximo Log-Return")

if len(df_lr) < 5:
    st.warning(
        "Quantidade insuficiente de dados para previsão confiável."
    )
else:
    ultimo_lr = df_lr["log_return"].iloc[-1]
    X_input = np.array([[ultimo_lr]])

    previsao = modelo.predict(X_input)[0]

    st.metric(
        label="Log-return previsto",
        value=f"{previsao:.6f}"
    )

st.caption("Tech Challenge • Fase 4 • FIAP")

