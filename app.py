# ============================================
# app.py - Dashboard Streamlit para Precificador Binomial
# Projeto Quant Researcher - Roda 100% na nuvem (Streamlit Cloud)
# ============================================

import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------------------------------------------------
# 1. CLASSE DO PRECIFICADOR (Coração do projeto)
# ------------------------------------------------------------
class BinomialTreePricer:
    """
    Precificador de Opções via Árvore Binomial (Modelo de Cox-Ross-Rubinstein).
    """
    def __init__(self, S, K, T, r, sigma, N, option_type='call', exercise='european'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()
        self.exercise = exercise.lower()
        
        # Discretização (Cox-Ross-Rubinstein)
        self.dt = T / N
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-r * self.dt)
        
    def price(self):
        """Calcula o preço justo via indução reversa."""
        # Árvore de preços no vencimento
        S_T = np.zeros(self.N + 1)
        for j in range(self.N + 1):
            S_T[j] = self.S * (self.u ** (self.N - j)) * (self.d ** j)
        
        # Payoff no vencimento
        V = np.zeros(self.N + 1)
        for j in range(self.N + 1):
            if self.option_type == 'call':
                V[j] = max(S_T[j] - self.K, 0)
            else:  # put
                V[j] = max(self.K - S_T[j], 0)
        
        # Indução reversa (backward induction)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                hold_value = self.discount * (self.p * V[j] + (1 - self.p) * V[j + 1])
                
                if self.exercise == 'american':
                    S_node = self.S * (self.u ** (i - j)) * (self.d ** j)
                    if self.option_type == 'call':
                        exercise_value = max(S_node - self.K, 0)
                    else:
                        exercise_value = max(self.K - S_node, 0)
                    V[j] = max(hold_value, exercise_value)
                else:
                    V[j] = hold_value
                    
        return V[0]

    def plot_convergence(self, max_N=150):
        """Plota a convergência do preço binomial para Black-Scholes."""
        # Fórmula de Black-Scholes
        def black_scholes(S, K, T, r, sigma):
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            if self.option_type == 'call':
                return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        bs_price = black_scholes(self.S, self.K, self.T, self.r, self.sigma)
        
        # Calcula os preços para diferentes N
        N_range = range(1, max_N + 1, 2)  # Ímpares para suavizar oscilações
        binomial_prices = []
        for n in N_range:
            pricer = BinomialTreePricer(
                self.S, self.K, self.T, self.r, self.sigma, n, 
                self.option_type, self.exercise
            )
            binomial_prices.append(pricer.price())
        
        # Cria o gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(N_range, binomial_prices, 'o-', 
                label='Preço da Árvore Binomial', markersize=3, color='darkblue')
        ax.axhline(y=bs_price, color='red', linestyle='--', 
                   label=f'Black-Scholes (Preço = {bs_price:.4f})')
        ax.set_xlabel('Número de Passos (N)')
        ax.set_ylabel('Preço da Opção')
        ax.set_title(f'Convergência: Opção {self.option_type.upper()} {self.exercise.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

# ------------------------------------------------------------
# 2. INTERFACE DO STREAMLIT (Dashboard)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Precificador de Opções - Árvore Binomial",
    page_icon="📈",
    layout="centered"
)

st.title("📊 Precificador de Opções por Árvore Binomial")
st.markdown("### Projeto para Portfólio de **Quant Researcher**")
st.markdown("---")

# Cria duas colunas para organizar os controles
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Parâmetros do Ativo")
    S = st.slider("Preço do Ativo (S)", min_value=50.0, max_value=200.0, value=100.0, step=5.0)
    K = st.slider("Preço de Exercício (K)", min_value=50.0, max_value=200.0, value=100.0, step=5.0)
    T = st.slider("Tempo até Vencimento (T - anos)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    r = st.slider("Taxa Livre de Risco (r)", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
    sigma = st.slider("Volatilidade (σ)", min_value=0.05, max_value=0.60, value=0.20, step=0.05)

with col2:
    st.subheader("⚙️ Configuração da Opção")
    N = st.slider("Número de Passos (N)", min_value=10, max_value=200, value=100, step=10)
    option_type = st.selectbox("Tipo de Opção", options=["call", "put"], index=0)
    exercise = st.selectbox("Estilo de Exercício", options=["european", "american"], index=0)
    st.caption("💡 Opções *Americanas* permitem exercício antecipado e geralmente são mais caras.")

st.markdown("---")

# Botão para disparar o cálculo (evita recalcular a cada movimento do slider)
if st.button("🚀 Calcular Preço Justo", use_container_width=True):
    with st.spinner("Construindo a árvore binomial e calculando..."):
        # Instancia o modelo
        pricer = BinomialTreePricer(S, K, T, r, sigma, N, option_type, exercise)
        preco = pricer.price()
        
        # 1. Exibe o resultado principal
        st.success(f"💰 **Preço Justo da Opção ({option_type.upper()} {exercise.upper()}): R$ {preco:.4f}**")
        
        # 2. Exibe o gráfico de convergência
        st.subheader("📉 Convergência para Black-Scholes")
        st.caption("Conforme N (passos) aumenta, o preço discreto da árvore converge para o modelo contínuo de Black-Scholes (linha vermelha).")
        fig = pricer.plot_convergence(max_N=150)
        st.pyplot(fig)
        plt.close(fig)  # Libera memória

st.markdown("---")
st.caption("Desenvolvido em Python com Streamlit | Projeto de Estudo para Quant Research")
