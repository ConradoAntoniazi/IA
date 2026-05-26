import numpy as np

N_IT = 100 # Num de iterações
N_INDIV = 1000 # Num de indivíduos
N_TUBOS = 10 # Num de tubos (dimensoes de cada individuo)

CROSS_OVER_TAX = 0.7
MUTATION_TAX = 0.1

trechos = {
    "comprimentos": np.array([20, 54, 98, 120, 34, 12, 88, 122, 33, 40]),
    "vazao": np.array([i * 2.5 for i in range(1, 11)])
}


tubos = {
    "diametro": np.array([150, 200, 250, 300, 400]),
    "custo": np.array([65, 98, 150, 210, 340])
}

# Criando as funções de vazao de fitness
def calcula_vazao_max_litros(diametro_mm):
    n = 0.013
    S = 0.005
    diametro_m = diametro_mm / 1000.0
    A = (np.pi * (diametro_m ** 2)) / 4
    Rh = diametro_m / 4
    Q_max = (1 / n) * A * (Rh ** (2 / 3)) * (S ** 0.5)
    return Q_max * 1000.0 # Converte para litros


def fitness(individuo):
    custo_total = 0
    penalidade = 0
    
    for i in range(N_TUBOS):
        idx_tubo = individuo[i]
        custo_total += trechos["comprimentos"][i] * tubos["custo"][idx_tubo]
        
        # Norma de segurança. Penalizar caso não atenda
        if trechos["vazao"][i] > calcula_vazao_max_litros(tubos["diametro"][idx_tubo]) * 0.75:
            penalidade += 100000
            
    return custo_total + penalidade
    
    
def selecao(pop):
    nova_pop = []
    
    while len(nova_pop) < N_INDIV:
        id_1 = np.random.randint(0, N_INDIV)
        id_2 = np.random.randint(0, N_INDIV)
        
        if fitness(pop[id_1]) < fitness(pop[id_2]):
            nova_pop.append(pop[id_1].copy())
            continue
        
        nova_pop.append(pop[id_2])
    
    return np.array(nova_pop)


def cross_over(pop):
    nova_pop = []
    
    while len(nova_pop) < N_INDIV:
        id_1 = np.random.randint(0, N_INDIV)
        id_2 = np.random.randint(0, N_INDIV)
        
        if np.random.uniform() < CROSS_OVER_TAX:
            idx = np.random.randint(0, N_TUBOS)
            filho_1 = np.concatenate([pop[id_1][:idx], pop[id_2][idx:]])
            filho_2 = np.concatenate([pop[id_2][:idx], pop[id_1][idx:]])
            
            nova_pop.append(filho_1.copy())
            nova_pop.append(filho_2.copy())
        
        else:
            nova_pop.append(pop[id_1])
            nova_pop.append(pop[id_2])
        
    return np.array(nova_pop)


def mutacao(pop):
    for i in range(N_INDIV):
        
        if np.random.uniform() <= MUTATION_TAX:
            ruido = np.random.randint(0, 2, size=N_TUBOS)
            pop[i] += ruido
            
            pop[i] = np.clip(pop[i], 0, 4)
            
            
# GA - Tubos
from tqdm import tqdm
import matplotlib.pyplot as plt

# Criando população inicial aleatória
pop = np.random.randint(low=0, high=5, size=(N_INDIV, N_TUBOS))
historico = []

for num in tqdm(range(N_IT), desc="Iterações do GA: "):
    pop = selecao(pop)
    pop = cross_over(pop)
    mutacao(pop)
    
    valor_hist = fitness(min(pop, key=fitness))
    
    historico.append(valor_hist)
    
    if len(historico) > 10:
        # Pega os 5 últimos elementos e joga em um conjunto (set)
        ultimos_cinco = set(historico[-10:])
        
        # Se o tamanho for 1, significa que todos os 5 são idênticos
        if len(ultimos_cinco) == 1:
            print("Evolução estagnou. Parando...")
            break
    
ind_min = min(pop, key=fitness)
print(ind_min)
print(f"Valor total (R$): {fitness(ind_min):.2f}")

plt.plot(historico)
plt.show()