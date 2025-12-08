# greenhorn_nav/utils/buoys2gates.py

import math
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional


def euclid(a, b) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def buoys2gates(
    red_dict: Dict,
    green_dict: Dict,
    max_match_distance: float = 5.0,
    sort_gates: bool = True,
    boat_position: Optional[np.ndarray] = None,
) -> Tuple[nx.Graph, List[Tuple], List[float]]:
    """Convert buoy dictionaries into gate matches, handling unpaired buoys.
    
    For unpaired buoys, creates virtual midpoints:
    - Unpaired RED: virtual midpoint offset LEFT (port side)
    - Unpaired GREEN: virtual midpoint offset RIGHT (starboard side)
    
    Args:
        red_dict: Red buoys {'r0': {'loc': [x,y]}, ...}
        green_dict: Green buoys {'g0': {'loc': [x,y]}, ...}
        max_match_distance: Max distance for red/green pair to form gate
        sort_gates: If True, sort by distance from boat_position
        boat_position: [x, y] boat position for sorting (defaults to origin)
    
    Returns:
        G: networkx.Graph with all buoys as nodes
        gate_matches: [(red_key, green_key, is_virtual), ...]
            - Paired: (red_key, green_key, False)
            - Unpaired red: (red_key, None, True)
            - Unpaired green: (None, green_key, True)
        scores: Match scores (distance for paired, 0.0 for virtual)
    """
    G = nx.Graph()
    
    # Add all buoys as nodes
    for k, v in (red_dict or {}).items():
        G.add_node(k, loc=list(v['loc']), color='red')
    for k, v in (green_dict or {}).items():
        G.add_node(k, loc=list(v['loc']), color='green')
    
    # Find paired gates using greedy matching
    gate_matches, scores, matched_reds, matched_greens = _match_paired_gates(
        red_dict, green_dict, max_match_distance, G
    )
    
    # Add unpaired buoys as virtual gates
    _add_virtual_gates(red_dict, green_dict, matched_reds, matched_greens, 
                      gate_matches, scores)
    
    # Sort gates by distance from boat
    if sort_gates and gate_matches and boat_position is not None:
        gate_matches, scores = _sort_gates_by_distance(
            gate_matches, scores, G, boat_position
        )
    
    return G, gate_matches, scores


def _match_paired_gates(red_dict: Dict, green_dict: Dict, 
                       max_match_distance: float, G: nx.Graph) -> Tuple:
    """Find paired gates using greedy closest-first matching."""
    matched_reds = set()
    matched_greens = set()
    gate_matches = []
    scores = []
    
    # Build candidate pairs sorted by distance
    candidates = []
    for rkey, rinfo in (red_dict or {}).items():
        for gkey, ginfo in (green_dict or {}).items():
            dist = euclid(rinfo['loc'], ginfo['loc'])
            if dist <= max_match_distance:
                candidates.append((dist, rkey, gkey))
    
    candidates.sort(key=lambda x: x[0])
    
    # Greedy matching: assign closest pairs first
    for dist, rkey, gkey in candidates:
        if rkey not in matched_reds and gkey not in matched_greens:
            gate_matches.append((rkey, gkey, False))
            scores.append(dist)
            matched_reds.add(rkey)
            matched_greens.add(gkey)
            G.add_edge(rkey, gkey, distance=dist)
    
    return gate_matches, scores, matched_reds, matched_greens


def _add_virtual_gates(red_dict: Dict, green_dict: Dict,
                      matched_reds: set, matched_greens: set,
                      gate_matches: List, scores: List):
    """Add unpaired buoys as virtual gates."""
    # Unpaired reds
    for rkey in (red_dict or {}).keys():
        if rkey not in matched_reds:
            gate_matches.append((rkey, None, True))
            scores.append(0.0)
    
    # Unpaired greens
    for gkey in (green_dict or {}).keys():
        if gkey not in matched_greens:
            gate_matches.append((None, gkey, True))
            scores.append(0.0)


def _sort_gates_by_distance(gate_matches: List, scores: List,
                           G: nx.Graph, boat_position: np.ndarray) -> Tuple:
    """Sort gates by distance from boat position."""
    def gate_distance(match):
        red_key, green_key, is_virtual = match
        
        if not is_virtual:
            # Paired: use midpoint
            red_loc = np.array(G.nodes[red_key]['loc'])
            green_loc = np.array(G.nodes[green_key]['loc'])
            midpoint = (red_loc + green_loc) / 2.0
        elif red_key:
            midpoint = np.array(G.nodes[red_key]['loc'])
        else:
            midpoint = np.array(G.nodes[green_key]['loc'])
        
        return np.linalg.norm(midpoint - boat_position)
    
    sorted_pairs = sorted(zip(gate_matches, scores), key=lambda x: gate_distance(x[0]))
    return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]


def compute_gate_midpoint_and_heading(
    red_loc: Optional[np.ndarray],
    green_loc: Optional[np.ndarray],
    next_midpoint: Optional[np.ndarray],
    prev_midpoint: Optional[np.ndarray],
    current_midpoint: Optional[np.ndarray],
    is_virtual: bool,
) -> Tuple[np.ndarray, float]:
    """Compute midpoint and heading for a gate (paired or virtual).
    
    Heading logic:
    - If next gate exists: point toward next gate's midpoint
    - If last gate and paired: perpendicular bisector pointing forward
    - If last gate and virtual: continue direction from previous gate
    
    Virtual gates:
    - Unpaired RED: virtual midpoint to LEFT of red buoy
    - Unpaired GREEN: virtual midpoint to RIGHT of green buoy
    
    Args:
        red_loc: Red buoy location [x, y] or None
        green_loc: Green buoy location [x, y] or None
        next_midpoint: Next gate's midpoint for heading calculation
        prev_midpoint: Previous gate's midpoint for fallback heading
        current_midpoint: Pre-computed offset midpoint (for virtual gates)
        is_virtual: True if unpaired gate
    
    Returns:
        (midpoint, heading) as (np.ndarray, float)
    """
    if not is_virtual:
        return _compute_paired_gate(red_loc, green_loc, next_midpoint)
    elif red_loc is not None:
        return _compute_virtual_red_gate(red_loc, current_midpoint, 
                                        next_midpoint, prev_midpoint)
    elif green_loc is not None:
        return _compute_virtual_green_gate(green_loc, current_midpoint,
                                          next_midpoint, prev_midpoint)
    else:
        raise ValueError("Both red_loc and green_loc cannot be None")


def _compute_paired_gate(red_loc: np.ndarray, green_loc: np.ndarray,
                        next_midpoint: Optional[np.ndarray]) -> Tuple:
    """Compute midpoint and heading for paired gate."""
    midpoint = (red_loc + green_loc) / 2.0
    
    if next_midpoint is not None:
        # Point toward next gate
        vec_to_next = next_midpoint - midpoint
        heading = math.atan2(vec_to_next[1], vec_to_next[0])
    else:
        # Last gate: use perpendicular bisector
        vec = red_loc - green_loc
        perp_vec = np.array([-vec[1], vec[0]])
        heading = math.atan2(perp_vec[1], perp_vec[0])
    
    return midpoint, heading


def _compute_virtual_red_gate(red_loc: np.ndarray, current_midpoint: np.ndarray,
                             next_midpoint: Optional[np.ndarray],
                             prev_midpoint: Optional[np.ndarray]) -> Tuple:
    """Compute midpoint and heading for unpaired red buoy (offset LEFT)."""
    heading = _compute_path_heading(current_midpoint, next_midpoint, prev_midpoint)
    return current_midpoint, heading


def _compute_virtual_green_gate(green_loc: np.ndarray, current_midpoint: np.ndarray,
                               next_midpoint: Optional[np.ndarray],
                               prev_midpoint: Optional[np.ndarray]) -> Tuple:
    """Compute midpoint and heading for unpaired green buoy (offset RIGHT)."""
    heading = _compute_path_heading(current_midpoint, next_midpoint, prev_midpoint)
    return current_midpoint, heading


def _compute_path_heading(current: np.ndarray,
                         next_mp: Optional[np.ndarray],
                         prev_mp: Optional[np.ndarray]) -> float:
    """Compute heading based on path direction."""
    if prev_mp is not None and next_mp is not None:
        # Middle gate: use overall path direction
        vec = next_mp - prev_mp
    elif prev_mp is not None:
        # End gate: continue from previous
        vec = current - prev_mp
    elif next_mp is not None:
        # Start gate: head toward next
        vec = next_mp - current
    else:
        # No context: default to east
        return 0.0
    
    return math.atan2(vec[1], vec[0])