# Planejamento de Investimentos em Marketing
'''
Uma startup está planejando a distribuição do orçamento mensal de marketing entre quatro canais de aquisição de clientes:
- Google Ads
- Instagram Ads
- LinkedIn Ads
- YouTube Ads
O objetivo da empresa é determinar quanto investir em cada canal para maximizar o retorno esperado da campanha, considerando que:
- cada canal possui eficiência diferente;
- investir demais em um mesmo canal gera retornos decrescentes;
- existem limitações estratégicas e orçamentárias.
'''

# Objetivo: Determinar quanto investir em cada canal para maximizar o retorno total da campanha

# Como as funções são todas do segundo grau, nao há muito risco de ótimos locais - optei por somente usar o hill climbing
import numpy as np


def funcao_lucro(sol):
    
    x1, x2, x3, x4 = sol
    
    ret_x1 = 50 * x1 - 1.2 * x1**2
    ret_x2 = 45 * x2 - 1.0 * x2**2
    ret_x3 = 40 * x3 - 0.8 * x3**2
    ret_x4 = 55 * x4 - 1.5 * x4**2
    
    return ret_x1 + ret_x2 + ret_x3 + ret_x4


def verifica_restricoes(sol):
    
    if min(sol) < 0:
        return False
    
    x1, x2, x3, x4 = sol
    
    rest_1 = np.sum(sol) <= 50.0
    
    rest_2 = x3 + x4 <= 25.0
    
    rest_3 = x1* 2 + x2*1 + x3*3 + x4*2 <= 80
    
    return rest_1 and rest_2 and rest_3


# trabalharei com 1 casa decimal
def gera_vizinhos(estado_atual):
    vizinhos = []
    
    for i in range(4):
        for delta in [-0.1, 0.1]:
            
            nova = list(estado_atual)
            nova[i] += round(nova[i] + delta, 1)

            if verifica_restricoes(nova):
                vizinhos.append(tuple(nova))

    return vizinhos
    

def hill_climbing(estado_inicial):
    estado_atual = estado_inicial
    
    while True:
        vizinhos = gera_vizinhos(estado_atual)
        
        if len(vizinhos) == 0:
            return estado_atual
        
        melhor = max(vizinhos, key=funcao_lucro) # Pega o melhor vizinho
        
        if funcao_lucro(estado_atual) >= funcao_lucro(melhor):
            return estado_atual
            
        estado_atual = melhor
        

estado_inicial = (0.0, 0.0, 0.0, 0.0)

resultado_final = hill_climbing(estado_inicial)

print("Resultado da otimização:")
print(f"\t- Invertimento em X1: {round(resultado_final[0], 1)}")
print(f"\t- Invertimento em X2: {round(resultado_final[1], 1)}")
print(f"\t- Invertimento em X3: {round(resultado_final[2], 1)}")
print(f"\t- Invertimento em X4: {round(resultado_final[3], 1)}")
print(f"\nLucro Gerado (R$): {funcao_lucro(resultado_final) * 1000}")