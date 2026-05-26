import numpy as np

# Declaração de constantes
MUTATION_FACTOR = 0.4
CROSS_FACTOR = 0.5
N_IND = 100
N_IT = 750
DIM = 30
INTERVAL = [-32, 32]

# Função a ser minimizada
def funcao_objetivo(x): 
    exp_1 = -0.2 * np.sqrt(1/DIM * np.sum(x**2))
    exp_2 = 1/DIM * np.sum(np.cos(2 * np.pi * x))
    
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + np.e

def mutacao(pop):
    pop_mutada = []
    
    while len(pop_mutada) < N_IND:
        id1, id2, id3 = np.random.randint(0, N_IND, (3))
        
        v = pop[id1] + MUTATION_FACTOR * (pop[id2] - pop[id3])
        v = np.clip(v, INTERVAL[0], INTERVAL[1])
        
        pop_mutada.append(v.copy())
            
    return np.array(pop_mutada)


def cross_over(pop_p, pop_m):
    pop_cruzada = []
    
    while len(pop_cruzada) < N_IND:
        id_p, id_m = np.random.randint(0, N_IND, (2))
        
        masc = np.random.uniform(size=DIM) < CROSS_FACTOR
        
        novo_ind = np.where(masc, pop_p[id_p], pop_m[id_m])
        
        pop_cruzada.append(novo_ind.copy())
        
    return np.array(pop_cruzada)


def selecao(pop):
    nova_pop = []
    
    while len(nova_pop) < N_IND:
        id1, id2 = np.random.randint(0, N_IND, (2))
        
        if funcao_objetivo(pop[id1]) < funcao_objetivo(pop[id2]):
            nova_pop.append(pop[id1].copy())
        else:
            nova_pop.append(pop[id2].copy())
            
    return np.array(nova_pop)


from tqdm import tqdm
import matplotlib.pyplot as plt

pop = np.random.uniform(INTERVAL[0], INTERVAL[1], size=(N_IND, DIM))
historico = []

for num in tqdm(range(N_IT), desc="Iterações do DE: "):
    pop_m = mutacao(pop)
    nova_pop = cross_over(pop, pop_m)
    pop = selecao(nova_pop)
    
    valor_hist = funcao_objetivo(min(pop, key=funcao_objetivo))
    
    historico.append(valor_hist)
    
    if len(historico) > 100:
        # Pega os 10 últimos elementos e joga em um conjunto (set)
        ultimos_cinco = set(historico[-100:])
        
        # Se o tamanho for 1, significa que todos os 10 são idênticos
        if len(ultimos_cinco) == 1:
            print("Evolução estagnou. Parando...")
            break
        
ind_min = min(pop, key=funcao_objetivo)
print(ind_min)
print(f"Valor: {funcao_objetivo(ind_min):.10f}")

plt.plot(historico)
plt.show()