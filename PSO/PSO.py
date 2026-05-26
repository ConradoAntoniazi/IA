import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

N_IND = 100
N_IT = 1000
DIM = 30
INTERVAL = [-32, 32]

W = 0.4
C1 = 2.05 
C2 = 2.05 

# CORREÇÃO 1: Adicionado 'axis' dinâmico para aceitar vetor 1D ou matriz 2D
def funcao_objetivo(x): 
    # Se for matriz (ndim == 2), soma nas colunas (axis=1). Se for vetor, soma tudo (axis=0)
    ax = 1 if x.ndim > 1 else 0
    exp_1 = -0.2 * np.sqrt(1/DIM * np.sum(x**2, axis=ax))
    exp_2 = 1/DIM * np.sum(np.cos(2 * np.pi * x), axis=ax)
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + np.e

class PSO():
    def __init__(self, dimension=30, n_indiv=100):
        self.n_indiv = n_indiv
        self.dimension = dimension
        
        self.pos = np.random.uniform(INTERVAL[0], INTERVAL[1], size=(n_indiv, dimension))
        self.vel = np.random.uniform(-1, 1, size=(n_indiv, dimension))
        self.pbest = self.pos.copy()
        
        fit_inicial = funcao_objetivo(self.pos)
        self.gbest = self.pos[np.argmin(fit_inicial)].copy()

    def atualiza_vel(self):
        r1 = np.random.uniform(size=(self.n_indiv, self.dimension))
        r2 = np.random.uniform(size=(self.n_indiv, self.dimension))
        self.vel = W * self.vel + C1 * r1 * (self.pbest - self.pos) + C2 * r2 * (self.gbest - self.pos)
        
    def atualiza_pos(self):
        self.pos += self.vel
        self.pos = np.clip(self.pos, INTERVAL[0], INTERVAL[1])
        
        for i in range(self.n_indiv):
            if funcao_objetivo(self.pos[i]) < funcao_objetivo(self.pbest[i]):
                self.pbest[i] = self.pos[i].copy()
                
        fit_pbest = funcao_objetivo(self.pbest)
        idx_gbest = np.argmin(fit_pbest)
        self.gbest = self.pbest[idx_gbest].copy()

        
pso = PSO(DIM, N_IND)
historico = []

for num in tqdm(range(N_IT), desc="Iterações do PSO: "):
    pso.atualiza_vel()
    pso.atualiza_pos()
    
    valor_hist = funcao_objetivo(pso.gbest)
    historico.append(valor_hist)
    
    if len(historico) >= 15:
        # Arredonda levemente para evitar problemas de precisão do float (ex: 0.0000001 != 0.0000002)
        ultimos = set(np.round(historico[-15:], 5))
        if len(ultimos) == 1:
            print("\nEvolução estagnou. Parando...")
            break
        
print("\nMelhor indivíduo:")
print(pso.gbest)
print(f"Melhor Valor: {funcao_objetivo(pso.gbest):.10f}")

plt.plot(historico)
plt.title("Convergência PSO")
plt.show()