# runTest.py - test e confronto soluzioni per gold thief
import sys
import os
import importlib
import pandas as pd
from pathlib import Path

# Aggiunge la directory principale al path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from Problem import Problem
from s346508 import solution  # il nostro solver
# Se vuoi usare funzioni utili come compute_cost, possiamo inserirle qui se necessario

def compute_cost(problem, path):
    """Calcola il costo di un percorso secondo alpha, beta e peso dell'oro."""
    alpha, beta = problem.alpha, problem.beta
    graph = problem.graph

    total_cost = 0.0
    w = 0.0
    for i in range(len(path)-1):
        current_city, gold_to_load = path[i]
        next_city = path[i+1][0]
        w += gold_to_load
        try:
            sp = nx.shortest_path(graph, current_city, next_city, weight='dist')
            d = nx.path_weight(graph, sp, weight='dist')
        except:
            return float('inf')
        total_cost += d + (d * alpha * w) ** beta
        if next_city == 0:
            w = 0.0
    return total_cost

def get_baseline_path(problem):
    """Percorsi di baseline: visita ogni città singolarmente."""
    total_path = []
    graph = problem.graph
    all_paths = nx.single_source_dijkstra_path(graph, source=0, weight='dist')
    for dest, path in all_paths.items():
        if dest == 0:
            continue
        gold_dest = graph.nodes[dest]['gold']
        total_path.extend([(0, 0), (dest, gold_dest), (0, 0)])
    return total_path

def run_tests():
    n_cities = [10, 50, 100]
    alpha_values = [0.0, 1.0, 2.0, 4.0]
    beta_values = [0.5, 1, 2, 4]
    density_values = [0.2, 0.5, 1.0]
    seed = 42

    configs = []
    for size in n_cities:
        for density in density_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    configs.append((size, density, alpha, beta))

    results = []
    for size, density, alpha, beta in configs:
        print(f"\nTesting: Size={size}, Density={density}, Alpha={alpha}, Beta={beta}")
        test_problem = Problem(size, density=density, alpha=alpha, beta=beta, seed=seed)

        baseline_path = get_baseline_path(test_problem)
        baseline_cost = compute_cost(test_problem, baseline_path)

        my_path = solution(test_problem)
        my_cost = compute_cost(test_problem, my_path)

        improvement = (baseline_cost - my_cost) / baseline_cost * 100
        print(f"  Baseline: {baseline_cost:.2f}, My: {my_cost:.2f}, Improvement: {improvement:.2f}%")

        results.append({
            'Size': size,
            'Density': density,
            'Alpha': alpha,
            'Beta': beta,
            'Baseline': f"{baseline_cost:.2f}",
            'Solution': f"{my_cost:.2f}",
            'Improvement%': f"{improvement:.2f}%"
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Salva i risultati in CSV dentro src/
    output_file = Path(__file__).parent / 'results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nRisultati salvati in {output_file}")

if __name__ == "__main__":
    import networkx as nx
    run_tests()
