# 📊 Precificador de Opções por Árvore Binomial (Cox-Ross-Rubinstein)

Aplicação interativa desenvolvida em **Python** e **Streamlit** para precificação de opções financeiras (Call/Put) de estilos **Europeu** e **Americano** utilizando a **Árvore Binomial de Cox-Ross-Rubinstein (CRR)** e validação de convergência empírica para a solução analítica de **Black-Scholes-Merton**.

---

## 🎯 Objetivo
Demonstrar a implementação prática de métodos numéricos para precificação de derivativos, permitindo a comparação direta do impacto do exercício antecipado (opções americanas) e a análise da convergência contínua à medida que o número de passos discretos ($N$) aumenta.

---

## 🛠️ Tecnologias Utilizadas
* **Python 3.x**
* **Streamlit** (Interface gráfica interativa)
* **NumPy** (Cálculos vetoriais de árvores e matrizes)
* **SciPy** (Distribuição normal acumulada para Black-Scholes)
* **Matplotlib** (Visualização gráfica de convergência)

---

## 📐 Fundamentação Teórica

### Modelo Binomial (Cox-Ross-Rubinstein)
A discretização do tempo até o vencimento $T$ em $N$ passos de tamanho $\Delta t = \frac{T}{N}$ define os fatores de alta ($u$) e baixa ($d$) do ativo subjacente:

$$u = e^{\sigma \sqrt{\Delta t}}, \quad d = \frac{1}{u}$$

A probabilidade de risco neutro $p$ e o fator de desconto são dados por:

$$p = \frac{e^{r \Delta t} - d}{u - d}, \quad \text{Desconto} = e^{-r \Delta t}$$

O cálculo do preço justo é obtido por **indução reversa (backward induction)** a partir dos *payoffs* no vencimento. Para opções americanas, avalia-se em cada nó a decisão de exercício antecipado:

$$V_{nó} = \max\left(\text{Valor de Manutenção}, \text{Valor de Exercício Antecipado}\right)$$

---

## 🚀 Como Executar o Projeto Localmente

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/WiseSeaHorse/Projeto.git](https://github.com/WiseSeaHorse/Projeto.git)
   cd Projeto
