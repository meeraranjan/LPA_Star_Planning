# greenhorn_nav/utils/buoys2gates.py

import math
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

def euclid(a, b):
    """Calculate Euclidean distance between two points."""
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))


def buoys2gates(
    red_dict: Dict,
    green_dict: Dict,
    max_match_distance: float = 5.0,
    sort_gates: bool = True,
    boat_position: Optional[np.ndarray] = None,
) -> Tuple[nx.Graph, List[Tuple], List[float]]:
    """
    Convert buoy dictionaries into gate matches, handling unpaired buoys.
    
    For unpaired buoys, creates virtual midpoints:
    - Unpaired RED buoys: virtual midpoint offset to the LEFT (port side)
    - Unpaired GREEN buoys: virtual midpoint offset to the RIGHT (starboard side)
    
    Args:
        red_dict: dict like {'r0': {'loc': [x,y]}, ...}
        green_dict: dict like {'g0': {'loc': [x,y]}, ...}
        max_match_distance: maximum distance to consider a red/green pair a gate
        sort_gates: if True, sort gates by distance from boat position
        boat_position: [x, y] position of boat for sorting gates (defaults to origin if None)
    
    Returns:
        G: networkx.Graph with nodes for all buoys
        gate_matches: list of tuples (red_key, green_key, is_virtual)
            - For paired gates: (red_key, green_key, False)
            - For unpaired red: (red_key, None, True)
            - For unpaired green: (None, green_key, True)
        scores: list of match scores (distance for paired, 0.0 for virtual)
    """
    G = nx.Graph()
    
    # Add all buoys as nodes
    for k, v in (red_dict or {}).items():
        G.add_node(k, loc=list(v['loc']), color='red')
    for k, v in (green_dict or {}).items():
        G.add_node(k, loc=list(v['loc']), color='green')
    
    # Track which buoys have been matched
    matched_reds = set()
    matched_greens = set()
    gate_matches = []
    scores = []
    
    # Phase 1: Compute all pairwise distances and create candidate pairs
    candidates = []  # List of (distance, red_key, green_key)
    for rkey, rinfo in (red_dict or {}).items():
        for gkey, ginfo in (green_dict or {}).items():
            d = euclid(rinfo['loc'], ginfo['loc'])
            if d <= max_match_distance:
                candidates.append((d, rkey, gkey))
    
    # Phase 2: Sort candidates by distance (greedy matching: closest pairs first)
    candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Greedily assign pairs (ensuring each buoy is used at most once)
    for dist, rkey, gkey in candidates:
        if rkey not in matched_reds and gkey not in matched_greens:
            gate_matches.append((rkey, gkey, False))  # False = not virtual
            scores.append(dist)
            matched_reds.add(rkey)
            matched_greens.add(gkey)
            G.add_edge(rkey, gkey, distance=dist)
    
    # Phase 4: Handle unpaired red buoys (create virtual midpoint to the LEFT)
    for rkey in red_dict.keys():
        if rkey not in matched_reds:
            gate_matches.append((rkey, None, True))  # True = virtual gate
            scores.append(0.0)
    
    # Phase 5: Handle unpaired green buoys (create virtual midpoint to the RIGHT)
    for gkey in green_dict.keys():
        if gkey not in matched_greens:
            gate_matches.append((None, gkey, True))  # True = virtual gate
            scores.append(0.0)
    
    # Phase 6: Sort gates by distance from boat position
    if sort_gates and gate_matches:
        # Default to origin if no boat position provided
        if boat_position is None:
            boat_position = np.array([0.0, 0.0])
        
        def gate_sort_key(match):
            red_key, green_key, is_virtual = match
            if not is_virtual:
                # Paired gate: use midpoint
                red_loc = np.array(G.nodes[red_key]['loc'])
                green_loc = np.array(G.nodes[green_key]['loc'])
                midpoint = (red_loc + green_loc) / 2.0
            elif red_key is not None:
                # Unpaired red
                midpoint = np.array(G.nodes[red_key]['loc'])
            else:
                # Unpaired green
                midpoint = np.array(G.nodes[green_key]['loc'])
            # Sort by distance from boat position
            return np.linalg.norm(midpoint - boat_position)
        
        # Sort gate_matches and scores together
        sorted_pairs = sorted(zip(gate_matches, scores), key=lambda x: gate_sort_key(x[0]))
        gate_matches = [pair[0] for pair in sorted_pairs]
        scores = [pair[1] for pair in sorted_pairs]
    
    return G, gate_matches, scores


def compute_gate_midpoint_and_heading(
    red_loc: Optional[np.ndarray],
    green_loc: Optional[np.ndarray],
    next_midpoint: Optional[np.ndarray],
    prev_midpoint: Optional[np.ndarray],
    current_midpoint: Optional[np.ndarray],
    is_virtual: bool,
) -> Tuple[np.ndarray, float]:
    """
    Compute midpoint and heading for a gate (paired or virtual).
    
    Heading logic:
    - If there's a next gate: point toward next gate's midpoint
    - If last gate and paired: perpendicular bisector pointing forward
    - If last gate and virtual: continue in direction from previous gate
    
    For virtual gates:
    - Unpaired RED buoy: virtual midpoint to the LEFT of red buoy
    - Unpaired GREEN buoy: virtual midpoint to the RIGHT of green buoy
    
    Args:
        red_loc: Location of red buoy [x, y] or None if unpaired green
        green_loc: Location of green buoy [x, y] or None if unpaired red
        next_midpoint: Location of next gate's midpoint [x, y] for heading calculation
        prev_midpoint: Location of previous gate's midpoint [x, y] for fallback heading
        current_midpoint: Pre-computed offset midpoint for this gate (for virtual gates)
        is_virtual: True if this is a virtual (unpaired) gate
    
    Returns:
        midpoint: [x, y] position of gate midpoint
        heading: angle in radians for boat to pass through gate
    """
    if not is_virtual:
        # Paired gate: midpoint is between red and green
        midpoint = (red_loc + green_loc) / 2.0

        # Heading calculation
        if next_midpoint is not None:
            # Point toward next gate
            vec_to_next = next_midpoint - midpoint
            heading = math.atan2(vec_to_next[1], vec_to_next[0])
        else:
            # Last gate: use perpendicular bisector pointing forward
            vec = red_loc - green_loc
            perp_vec = np.array([-vec[1], vec[0]])
            heading = math.atan2(perp_vec[1], perp_vec[0])

    elif red_loc is not None:
        # Unpaired red buoy: virtual midpoint to the LEFT
        # Use current_midpoint (pre-computed offset position) for heading calculations
        if prev_midpoint is not None and next_midpoint is not None:
            # Middle gate: use path direction
            vec_path = next_midpoint - prev_midpoint
        elif prev_midpoint is not None:
            # End gate: continue from previous
            vec_path = current_midpoint - prev_midpoint
        elif next_midpoint is not None:
            # Start gate: head toward next
            vec_path = next_midpoint - current_midpoint
        else:
            vec_path = None

        if vec_path is not None:
            heading = math.atan2(vec_path[1], vec_path[0])
        else:
            heading = 0.0

        # Use the pre-computed offset midpoint
        midpoint = current_midpoint

    elif green_loc is not None:
        # Unpaired green buoy: virtual midpoint to the RIGHT
        # Use current_midpoint (pre-computed offset position) for heading calculations
        if prev_midpoint is not None and next_midpoint is not None:
            # Middle gate: use path direction
            vec_path = next_midpoint - prev_midpoint
        elif prev_midpoint is not None:
            # End gate: continue from previous
            vec_path = current_midpoint - prev_midpoint
        elif next_midpoint is not None:
            # Start gate: head toward next
            vec_path = next_midpoint - current_midpoint
        else:
            vec_path = None

        if vec_path is not None:
            heading = math.atan2(vec_path[1], vec_path[0])
        else:
            heading = 0.0

        # Use the pre-computed offset midpoint
        midpoint = current_midpoint

    else:
        raise ValueError("Both red_loc and green_loc cannot be None")
    
    return midpoint, heading