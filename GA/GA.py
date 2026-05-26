import numpy as np

DIMENSAO = 30
INDIVIDUOS = 150
EPOCAS = 500
INTERVALO = [-32, 32]

TAXA_CRUZAMENTO = 0.99
TAXA_MUTACAO = 0.01

def fitness(x):
    exp_1 = -0.2 * np.sqrt(np.mean(x**2))
    exp_2 = np.mean(np.cos(2 * np.pi * x))
    
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + np.e


def selecao(pop):
    pop_s = []
    tam_pop = len(pop)
    
    for _ in range(tam_pop):
        
        id1, id2 = np.random.randint(0, tam_pop, size=(2))
        
        if fitness(pop[id1]) < fitness(pop[id2]):
            pop_s.append(pop[id1].copy())
        else:
            pop_s.append(pop[id2].copy())
            
    return np.array(pop_s)


def crossOver(pop, D=DIMENSAO, c_tax=TAXA_CRUZAMENTO):
    pop_co = []
    tam_pop = len(pop)
    
    while len(pop_co) < tam_pop:
        id1, id2 = np.random.randint(0, tam_pop, size=2)
        
        if np.random.uniform() < c_tax:
            separate = np.random.randint(1, D)
            f1 = np.concatenate((pop[id1][:separate], pop[id2][separate:]))
            f2 = np.concatenate((pop[id2][:separate], pop[id1][separate:]))
            
            pop_co.append(f1)
            pop_co.append(f2)
            
        else:
            pop_co.append(pop[id1].copy())
            pop_co.append(pop[id2].copy())
            
    return np.array(pop_co)


def mutacao(pop, D=DIMENSAO, m_tax=TAXA_MUTACAO, interval=INTERVALO):
    pop_m = []
    
    for p in pop:
        ruido = np.random.uniform(size=D)
        masc = ruido < m_tax
        
        p = np.where(masc, p, np.random.uniform(-1, 1))
        pop_m.append(p)
        
    return pop_m
        
        
from tqdm import tqdm
import matplotlib.pyplot as plt

pop = np.random.uniform(INTERVALO[0], INTERVALO[1], size=(INDIVIDUOS, DIMENSAO))
historico = []

for num in tqdm(range(EPOCAS), desc="Iterações do GA: "):
    pop_s = selecao(pop)         # Seleciona os melhores
    nova_pop = crossOver(pop_s)  # Faz crossover baseado nos selecionados (pop_s)
    pop = mutacao(nova_pop)      # Aplica a mutação na nova geração
    
    valor_hist = fitness(min(pop, key=fitness))
    
    historico.append(valor_hist)
    
    if len(historico) > 100:
        # Pega os 10 últimos elementos e joga em um conjunto (set)
        ultimos_cinco = set(historico[-100:])
        
        # Se o tamanho for 1, significa que todos os 10 são idênticos
        if len(ultimos_cinco) == 1:
            print("Evolução estagnou. Parando...")
            break
        
ind_min = min(pop, key=fitness)
print(ind_min)
print(f"Valor: {fitness(ind_min):.10f}")

plt.plot(historico)
plt.show()