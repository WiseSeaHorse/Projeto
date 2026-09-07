import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Classe do Precificador (idêntica à anterior) ---
class BinomialTreePricer:
    def __init__(self, S, K, T, r, sigma, N, option_type='call', exercise='european'):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma; self.N = N
        self.option_type = option_type.lower(); self.exercise = exercise.lower()
        self.dt = T / N
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-r * self.dt)
        
    def price(self):
        S_T = np.zeros(self.N + 1)
        for j in range(self.N + 1):
            S_T[j] = self.S * (self.u ** (self.N - j)) * (self.d ** j)
        V = np.zeros(self.N + 1)
        for j in range(self.N + 1):
            V[j] = max(S_T[j] - self.K, 0) if self.option_type == 'call' else max(self.K - S_T[j], 0)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                hold_value = self.discount * (self.p * V[j] + (1 - self.p) * V[j + 1])
                if self.exercise == 'american':
                    S_node = self.S * (self.u ** (i - j)) * (self.d ** j)
                    exercise_value = max(S_node - self.K, 0) if self.option_type == 'call' else max(self.K - S_node, 0)
                    V[j] = max(hold_value, exercise_value)
                else:
                    V[j] = hold_value
        return V[0]

    def plot_convergence(self, max_N=100):
        def black_scholes(S, K, T, r, sigma):
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            if self.option_type == 'call':
                return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        bs_price = black_scholes(self.S, self.K, self.T, self.r, self.sigma)
        N_range = range(1, max_N + 1, 2)
        binomial_prices = []
        for n in N_range:
            pricer = BinomialTreePricer(self.S, self.K, self.T, self.r, self.sigma, n, self.option_type, self.exercise)
            binomial_prices.append(pricer.price())
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(N_range, binomial_prices, 'o-', label='Árvore Binomial', markersize=4)
        ax.axhline(y=bs_price, color='r', linestyle='--', label=f'Black-Scholes (Preço = {bs_price:.4f})')
        ax.set_xlabel('Número de Passos (N)'); ax.set_ylabel('Preço da Opção')
        ax.set_title(f'Convergência - {self.option_type.upper()} {self.exercise.upper()}')
        ax.legend(); ax.grid(True)
        return fig

# --- INTERFACE DO STREAMLIT (Tudo na nuvem) ---
st.set_page_config(page_title="Precificador de Opções - Quant Researcher", layout="wide")
st.title("📊 Precificador de Opções por Árvore Binomial")
st.markdown("### Projeto para Portfólio de Quant Research")

# Colunas para organizar os controles
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Parâmetros do Ativo")
    S = st.slider("Preço do Ativo (S)", 50.0, 200.0, 100.0, 5.0)
    K = st.slider("Preço de Exercício (K)", 50.0, 200.0, 100.0, 5.0)
    T = st.slider("Tempo até Vencimento (T - anos)", 0.1, 3.0, 1.0, 0.1)
    r = st.slider("Taxa Livre de Risco (r)", 0.0, 0.20, 0.05, 0.01)
    sigma = st.slider("Volatilidade (σ)", 0.05, 0.60, 0.20, 0.05)

with col2:
    st.subheader("⚙️ Configuração da Opção")
    N = st.slider("Número de Passos (N)", 10, 200, 100, 10)
    option_type = st.selectbox("Tipo de Opção", ["call", "put"])
    exercise = st.selectbox("Estilo de Exercício", ["european", "american"])
    st.caption("💡 *Americanas permitem exercício antecipado e geralmente são mais caras.*")

# Botão para recalcular (opcional, mas evita travamentos)
if st.button("🚀 Calcular Preço e Gerar Gráfico"):
    with st.spinner("Calculando árvore binomial..."):
        pricer = BinomialTreePricer(S, K, T, r, sigma, N, option_type, exercise)
        preco = pricer.price()
        
        # Mostra o resultado em destaque
        st.success(f"💰 Preço Justo da Opção: **R$ {preco:.4f}**")
        
        # Mostra o gráfico
        st.subheader("📉 Convergência para Black-Scholes")
        fig = pricer.plot_convergence(max_N=150)
        st.pyplot(fig)
        plt.close(fig)
