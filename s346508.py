# s346508.py - Gold Thief Solver
# Heuristica greedy incrementale con merge adattivo e rumore controllato

import networkx as nx
import numpy as np
import random
import time


def goldThief(problem, seed: int = 42):
    """
    Heuristica incrementale:
    - costruisce inizialmente viaggi elementari
    - fonde solo catene con beneficio netto misurabile
    - limita la crescita delle catene in funzione di beta
    - introduce rumore controllato per evitare stagnazione
    """
    random.seed(seed)
    graph = problem.graph
    alpha, beta = problem.alpha, problem.beta
    start_time = time.time()

    # --- Preparazione nodi ---
    cities = [n for n in graph.nodes if n != 0]
    n_targets = len(cities)

    city_to_idx = {c: i for i, c in enumerate(cities)}
    idx_to_city = {i: c for i, c in enumerate(cities)}
    depot_idx = n_targets

    # --- Matrice delle distanze minime ---
    dist = np.full((n_targets + 1, n_targets + 1), np.inf, dtype=np.float32)
    mapping = city_to_idx.copy()
    mapping[0] = depot_idx

    for u, dists in nx.all_pairs_dijkstra_path_length(graph, weight="dist"):
        if u not in mapping:
            continue
        ui = mapping[u]
        for v, d in dists.items():
            if v in mapping:
                dist[ui, mapping[v]] = d

    # --- Oro ---
    gold_dict = nx.get_node_attributes(graph, "gold")
    gold_values = np.array([gold_dict[idx_to_city[i]] for i in range(n_targets)])

    # --- Funzione costo di una catena ---
    def compute_route_cost(route):
        total_cost = 0.0
        carried = 0.0
        current = depot_idx

        for idx in route:
            d = dist[current, idx]
            total_cost += d + (d * alpha * carried) ** beta
            carried += gold_values[idx]
            current = idx

        d_back = dist[current, depot_idx]
        total_cost += d_back + (d_back * alpha * carried) ** beta
        return total_cost

    # --- Inizializzazione: una città per viaggio ---
    routes = {i: [i] for i in range(n_targets)}
    route_costs = {i: compute_route_cost([i]) for i in range(n_targets)}
    city_owner = {i: i for i in range(n_targets)}

    # --- Vicini candidati (randomizzati) ---
    neighbors = {}
    for i in range(n_targets):
        close = np.argsort(dist[i, :n_targets])[1:12]
        neighbors[i] = list(close)
        random.shuffle(neighbors[i])

    # --- Ciclo di miglioramento ---
    MAX_ITER = 60
    TIME_LIMIT = 25
    improved = True
    iteration = 0

    while improved and iteration < MAX_ITER and (time.time() - start_time) < TIME_LIMIT:
        improved = False
        iteration += 1

        best_gain = 1e-5
        best_choice = None

        for rid_a in list(routes.keys()):
            if rid_a not in routes:
                continue

            route_a = routes[rid_a]
            last_city = route_a[-1]

            for candidate in neighbors[last_city]:
                rid_b = city_owner[candidate]

                if rid_a == rid_b or rid_b not in routes:
                    continue

                route_b = routes[rid_b]
                if route_b[0] != candidate:
                    continue

                # Limite dinamico della lunghezza delle catene
                max_len = 2 + int(beta // 1.5)
                if beta < 2:
                    max_len += 1

                if len(route_a) + len(route_b) > max_len:
                    continue

                merged = route_a + route_b
                new_cost = compute_route_cost(merged)
                old_cost = route_costs[rid_a] + route_costs[rid_b]
                gain = old_cost - new_cost

                # Soglia adattiva con rumore controllato
                threshold = 1e-5 * old_cost * (1 + 0.2 * random.random())

                if gain > best_gain and gain > threshold:
                    best_gain = gain
                    best_choice = (rid_a, rid_b, merged, new_cost)

        if best_choice:
            ra, rb, merged, new_cost = best_choice
            routes[ra] = merged
            route_costs[ra] = new_cost

            for node in routes[rb]:
                city_owner[node] = ra

            del routes[rb]
            del route_costs[rb]
            improved = True

    # --- Costruzione percorso finale (ordine per costo decrescente) ---
    final_path = [(0, 0)]
    for rid in sorted(routes, key=lambda r: route_costs[r], reverse=True):
        for idx in routes[rid]:
            city = idx_to_city[idx]
            final_path.append((city, gold_dict[city]))
        final_path.append((0, 0))

    return final_path
