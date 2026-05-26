import numpy as np

DIM = 30
INTERVALO = [-32, 32]


def funcao(x):
    x = np.array(x)
    
    exp_1 = -0.2 * np.sqrt(1.0 / DIM * np.sum(np.square(x)))
    
    exp_2 = 1.0 / DIM * np.sum(np.cos(2 * np.pi * x))
    
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + np.e


def verifica_retricoes(sol):
    return np.all(sol >= INTERVALO[0]) and np.all(sol <= INTERVALO[1])


def gera_vizinhos_hc(sol_atual):
    vizinhos = []
    
    for i in range(DIM):
        for delta in [-0.1, 0.1]:
            nova = sol_atual.copy()
            nova[i] += delta
            
            if verifica_retricoes(nova):
                vizinhos.append(nova)
                
    return vizinhos


def hill_climbing(sol_inicial):
    sol_atual = sol_inicial.copy()
    
    while True:
        vizinhos = gera_vizinhos_hc(sol_atual)
        
        if len(vizinhos) == 0:
            break

        minimo = min(vizinhos, key=funcao)
        
        if funcao(sol_atual) <= funcao(minimo):
            break
        
        sol_atual = minimo
        
    return sol_atual


def gera_vizinho_sa(atual):
    while True:
        ruido = np.random.normal(0, 1, DIM)
        
        novo_vizinho = atual + ruido
        
        if verifica_retricoes(novo_vizinho):
            return novo_vizinho


def simulated_annealing(sol_inicial):
    atual = sol_inicial.copy()
    melhor = atual.copy()

    T = 100.0
    T_min = 0.1
    alpha = 0.9999

    while T > T_min:
        vizinho_aleatorio = gera_vizinho_sa(atual)

        delta = funcao(vizinho_aleatorio) - funcao(atual)

        if delta < 0:
            atual = vizinho_aleatorio.copy()
        elif np.random.uniform() < np.exp(-delta / T):
            atual = vizinho_aleatorio.copy()

        if funcao(atual) < funcao(melhor):
            melhor = atual.copy()

        T *= alpha

    return melhor


# Teste Real iniciando de uma posição aleatória longe do ótimo
sol_inicial = np.random.uniform(INTERVALO[0], INTERVALO[1], DIM)

print("--- Executando Simulated Annealing ---")
melhor_sa = simulated_annealing(sol_inicial)
print(f"Custo após SA: {funcao(melhor_sa):.4f}")

print("\n--- Refinando o resultado com Hill Climbing ---")
resultado_final = hill_climbing(melhor_sa)
print(f"Custo Final após HC: {funcao(resultado_final):.4f}")
print(f"Distância média do centro (0,0...): {np.mean(np.abs(resultado_final)):.4f}")