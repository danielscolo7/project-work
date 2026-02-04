# s346508.py
from goldThief import goldThief  

def solution(problem):
    """
    Wrapper richiesto dalle istruzioni:
    prende l'istanza Problem e restituisce il percorso ottimale,
    delegando la logica alla funzione goldThief.
    """
    # Estrazione dei parametri principali da Problem
    alpha = problem.alpha
    beta = problem.beta
    seed = 42
    max_iter = 60
    time_limit = 25
    neighbor_count = 12

    return goldThief(problem, alpha=alpha, beta=beta, seed=seed,
                     max_iter=max_iter, time_limit=time_limit,
                     neighbor_count=neighbor_count)


