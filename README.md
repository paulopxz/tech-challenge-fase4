# Tech Challenge – Fase 4  
## Previsão do Ibovespa com ARIMA

Este projeto apresenta uma aplicação desenvolvida em **Streamlit** para
visualização e previsão do Ibovespa, como parte da **Fase 4 do Tech Challenge**.

---

## 📊 Dados

Os dados históricos do Ibovespa (2005–2025) são carregados a partir de um arquivo
CSV contendo informações diárias de mercado.

A coluna de preço utilizada é **"Último"**, convertida para formato numérico e
tratada conforme o padrão brasileiro.

---

## 🧠 Modelo

O modelo utilizado é um **ARIMA**, definido e validado durante a **Fase 2** do
projeto, treinado sobre os **log-retornos do Ibovespa**.

Devido a limitações de portabilidade de modelos do `statsmodels` entre ambientes
(distintas versões de NumPy e bibliotecas no Streamlit Cloud), o modelo é
**reajustado dinamicamente dentro da aplicação**.

Essa abordagem garante:
- compatibilidade no deploy em nuvem
- previsões consistentes
- aderência ao modelo definido na Fase 2

---

## 🚀 Aplicação

A aplicação permite:
- visualização da série histórica do Ibovespa
- cálculo de log-retornos
- previsão do próximo log-retorno via ARIMA

---

## 📦 Estrutura do Projeto

tech-challenge-fase4/
│
├── app.py
├── README.md
├── requirements.txt
├── data/
│ └── Dados Históricos - Ibovespa 2005-2025.csv
└── notebook/
└── Tech_challenge_fase_2_para_fase_4.ipynb

## 🚀 Como Executar Localmente

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute a aplicação:
   ```bash
   streamlit run app_.py
   ```

---

## 🌐 Deploy

O deploy da aplicação foi realizado utilizando o **Streamlit Cloud**, com
integração direta ao repositório do GitHub.

---

## 📹 Vídeo Demonstrativo

Foi produzido um vídeo de até **5 minutos**, apresentando:

- O contexto do problema
- O modelo desenvolvido na Fase 2
- A aplicação Streamlit em funcionamento
- O painel de métricas e monitoramento

---

## 👨‍🎓 Projeto Acadêmico

Projeto desenvolvido para fins acadêmicos no curso **POSTECH – FIAP**,
como parte do **Tech Challenge – Fase 4**.


## ✅ Observação Final

A estratégia adotada é uma prática comum em projetos de séries temporais
em produção, priorizando estabilidade e reprodutibilidade do modelo
em ambientes de deploy.
