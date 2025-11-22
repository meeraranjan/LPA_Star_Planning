# greenhorn_nav/utils/gate_computation.py

import numpy as np
from typing import List, Optional
from .gate_state import Gate
from .buoys2gates import buoys2gates, compute_gate_midpoint_and_heading


class GateBuilder:
    """Builds gate sequences from buoy data with proper midpoint/heading calculation."""
    
    def __init__(self, unpaired_red_offset: float = 0.5, 
        unpaired_green_offset: float = 0.5,
        max_gate_match_distance: float = 5.0):
        self.unpaired_red_offset = unpaired_red_offset
        self.unpaired_green_offset = unpaired_green_offset
        self.max_gate_match_distance = max_gate_match_distance
    
    def build_gates(self, red_dict: dict, green_dict: dict,
            boat_pos: np.ndarray,
            passed_midpoints: List[np.ndarray]) -> List[Gate]:
        """Build gate sequence from buoy dictionaries.
        
        Args:
            red_dict: Red buoy locations {key: {'loc': [x, y]}}
            green_dict: Green buoy locations {key: {'loc': [x, y]}}
            boat_pos: Current boat position [x, y]
            passed_midpoints: Midpoints of previously passed gates
            
        Returns:
            List of Gate objects in sequence
        """
        if not red_dict and not green_dict:
            return []
        
        # Get gate matches
        G, gate_matches, _ = buoys2gates(
            red_dict, green_dict,
            max_match_distance=self.max_gate_match_distance,
            boat_position=boat_pos
        )
        
        if not gate_matches:
            return []
        
        # Extract gate info
        gate_info = []
        for red_key, green_key, is_virtual in gate_matches:
            red_loc = np.array(G.nodes[red_key]['loc']) if red_key else None
            green_loc = np.array(G.nodes[green_key]['loc']) if green_key else None
            gate_info.append((red_loc, green_loc, is_virtual, red_key, green_key))
        
        # First pass: compute preliminary midpoints (needed for heading calculation)
        prelim_midpoints = self._compute_preliminary_midpoints(
            gate_info, passed_midpoints, boat_pos
        )
        
        # Second pass: compute final midpoints and headings
        gates = []
        for i, (red_loc, green_loc, is_virtual, red_key, green_key) in enumerate(gate_info):
            idx = len(passed_midpoints) + i
            
            next_mp = prelim_midpoints[idx + 1] if idx + 1 < len(prelim_midpoints) else None
            prev_mp = prelim_midpoints[idx - 1] if idx > 0 else None
            current_mp = prelim_midpoints[idx]
            
            midpoint, heading = compute_gate_midpoint_and_heading(
                red_loc=red_loc,
                green_loc=green_loc,
                next_midpoint=next_mp,
                prev_midpoint=prev_mp,
                current_midpoint=current_mp,
                is_virtual=is_virtual
            )
            
            gate = Gate(
                midpoint=midpoint,
                heading=heading,
                is_virtual=is_virtual,
                red_key=red_key,
                green_key=green_key
            )
            gates.append(gate)
        
        return gates
    
    def _compute_preliminary_midpoints(self, gate_info: list,
                                    passed_midpoints: List[np.ndarray],
                                    boat_pos: np.ndarray) -> List[np.ndarray]:
        """Compute all preliminary midpoints for heading estimation."""
        all_midpoints = passed_midpoints.copy()
        boat_heading = 0.0  # Default fallback
        
        for i, (red_loc, green_loc, is_virtual, _, _) in enumerate(gate_info):
            idx = len(passed_midpoints) + i
            
            if not is_virtual:
                # Paired gate: simple midpoint
                midpoint = (red_loc + green_loc) / 2.0
            
            elif red_loc is not None:
                # Unpaired red: offset LEFT
                heading = self._estimate_heading(all_midpoints, idx, red_loc, boat_heading)
                left_angle = heading + np.pi / 2
                offset = self.unpaired_red_offset * np.array([np.cos(left_angle), 
                                                            np.sin(left_angle)])
                midpoint = red_loc + offset
            
            else:
                # Unpaired green: offset RIGHT
                heading = self._estimate_heading(all_midpoints, idx, green_loc, boat_heading)
                right_angle = heading - np.pi / 2
                offset = self.unpaired_green_offset * np.array([np.cos(right_angle), 
                                                            np.sin(right_angle)])
                midpoint = green_loc + offset
            
            all_midpoints.append(midpoint)
        
        return all_midpoints
    
    def _estimate_heading(self, midpoints: List[np.ndarray], 
                        current_idx: int, buoy_loc: np.ndarray,
                        fallback_heading: float) -> float:
        """Estimate heading for virtual gate offset calculation."""
        if current_idx > 0:
            vec = buoy_loc - midpoints[-1]
            return np.arctan2(vec[1], vec[0])
        return fallback_heading