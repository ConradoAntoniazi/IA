import numpy as np

COEF_INERCIA = 0.6
COMP_COGN = 2.05
COMP_SOC = 2.05

DIMENSOES = 30
TAM_POP = 100
VEL_MIN = -0.5
VEL_MAX = 0.5
POS_MIN = -32
POS_MAX = 32

def fitness(individuo_pos):
    exp_1 = -0.2 * np.sqrt(1/DIMENSOES * np.sum(individuo_pos**2))
    exp_2 = 1/DIMENSOES * np.sum(np.cos(2 * np.pi * individuo_pos))
    
    return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + np.e

class Particle:
    def __init__(self, velocidade, posicao):
        self.velocidade = velocidade
        self.posicao = posicao
        self.pbest = np.copy(posicao)
        self.gbest = np.copy(posicao)

    def atualiza_velocidade(self, w, c1, c2):
        r1 = np.random.uniform()
        r2 = np.random.uniform()

        v = w*self.velocidade + c1*r1*(self.pbest - self.posicao) + c2*r2*(self.gbest - self.posicao)
        v = np.clip(v, VEL_MIN, VEL_MAX)
        
        self.velocidade = v

    def atualiza_posicao(self):
        p = self.posicao + self.velocidade
        p = np.clip(a=p, a_min=POS_MIN, a_max=POS_MAX)
        
        self.posicao = p


def random_particle(vmin=0, vmax=0.5, pmin=-32, pmax=32):
    vel = np.random.uniform(low=vmin,
                            high=vmax)
    pos = np.random.uniform(low=pmin,
                            high=pmax,
                            size=DIMENSOES)
    
    return Particle(vel, pos)
    
def init_pop():
    return np.array([random_particle(VEL_MIN, VEL_MAX, POS_MIN, POS_MAX) for _ in range(TAM_POP)])

def pso(max_iter=250):
    pop = init_pop()

    for i in range(max_iter):
        idx_best_particle = np.argmin([fitness(p.posicao) for p in pop])
        best_particle = pop[idx_best_particle]

        if fitness(best_particle.posicao) < fitness(best_particle.gbest):
            for p in pop:
                p.gbest = np.copy(best_particle.posicao)
        
        for p in pop:
            if fitness(p.posicao) < fitness(p.pbest):
                p.pbest = np.copy(p.posicao)

            p.atualiza_velocidade(COEF_INERCIA, COMP_COGN, COMP_SOC)
            p.atualiza_posicao()
        
        # print(f"Iteration {i+1}/{max_iter}, Best Fitness = {fitness(best_particle.gbest):.6f}")


    return best_particle.posicao, best_particle.gbest

def main():
    melhor_pos, melhor_gbest = pso(1500)

    print("\nOptimal Solution Found:")
    print("Best Position:", melhor_pos)
    print("Best Fitness:", fitness(melhor_gbest))

if __name__ == "__main__":
    main()