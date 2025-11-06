# greenhorn_mapping/utils.py  (dummy implementation)
import math
import networkx as nx
import numpy as np

def euclid(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

def buoys2gates(red_dict, green_dict, max_match_distance=5.0):
    """
    Dummy converter from buoys -> gates.

    Args:
        red_dict: dict like {'r0': {'loc': [x,y]}, ...}
        green_dict: dict like {'g0': {'loc': [x,y]}, ...}
        max_match_distance: maximum distance to consider a red/green pair a gate

    Returns:
        G: networkx.Graph with nodes for all buoys; node attribute 'loc' holds [x,y]
        gate_matches: list of (red_key, green_key) pairs
        scores: list of match scores (distance)
    """
    G = nx.Graph()

    # Add all buoys as nodes with loc attribute
    for k, v in (red_dict or {}).items():
        G.add_node(k, loc=list(v['loc']), color='red')
    for k, v in (green_dict or {}).items():
        G.add_node(k, loc=list(v['loc']), color='green')

    # Match each red to the closest green (if within threshold)
    gate_matches = []
    scores = []
    for rkey, rinfo in (red_dict or {}).items():
        best_g = None
        best_dist = float('inf')
        for gkey, ginfo in (green_dict or {}).items():
            d = euclid(rinfo['loc'], ginfo['loc'])
            if d < best_dist:
                best_dist = d
                best_g = gkey
        if best_g is not None and best_dist <= max_match_distance:
            gate_matches.append((rkey, best_g))
            scores.append(best_dist)
            # Optionally add an edge in graph for visualization
            G.add_edge(rkey, best_g, distance=best_dist)
    return G, gate_matches, scores
