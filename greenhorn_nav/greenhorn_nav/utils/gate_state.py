# greenhorn_nav/utils/gate_state.py

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field


@dataclass
class Gate:
    """Represents a single gate with its midpoint, heading, and metadata."""
    midpoint: np.ndarray
    heading: float
    is_virtual: bool
    red_key: Optional[str] = None
    green_key: Optional[str] = None
    
    @property
    def gate_id(self) -> Tuple:
        """Unique identifier for this gate."""
        return (tuple(np.round(self.midpoint, 3)), self.red_key, self.green_key)


class GateManager:
    """Manages gate state, progression, and buoy tracking."""
    
    def __init__(self, waypoint_pass_tol: float = 0.07):
        self.gates: List[Gate] = []
        self.current_idx: int = 0
        self.waypoint_pass_tol = waypoint_pass_tol
        
        # Track what's been used/passed
        self.used_red_buoys: Set[str] = set()
        self.used_green_buoys: Set[str] = set()
        self.passed_gate_ids: Set[Tuple] = set()
        self.passed_midpoints: List[np.ndarray] = []
    
    def update_gates(self, new_gates: List[Gate]):
        """Replace current gates with new sequence."""
        self.gates = new_gates
        self.current_idx = 0
    
    def get_current_gate(self) -> Optional[Gate]:
        """Get the gate boat should currently navigate to."""
        if not self.gates:
            return None
        return self.gates[min(self.current_idx, len(self.gates) - 1)]
    
    def check_and_advance(self, boat_x: float, boat_y: float, boat_heading: float) -> bool:
        """Check if current gate is passed and advance if so.
        
        Returns:
            True if gate was just passed, False otherwise
        """
        if self.current_idx >= len(self.gates):
            return False
        
        gate = self.gates[self.current_idx]
        
        # Already passed this gate?
        if gate.gate_id in self.passed_gate_ids:
            return False
        
        # Check if passed
        if not self._is_gate_passed(gate.midpoint, boat_x, boat_y, boat_heading):
            return False
        
        # Mark as passed
        self.passed_gate_ids.add(gate.gate_id)
        self.passed_midpoints.append(gate.midpoint.copy())
        
        if gate.red_key:
            self.used_red_buoys.add(gate.red_key)
        if gate.green_key:
            self.used_green_buoys.add(gate.green_key)
        
        # Advance to next gate if not at end
        if self.current_idx < len(self.gates) - 1:
            self.current_idx += 1
        
        return True
    
    def _is_gate_passed(self, gate_midpoint: np.ndarray, 
                       boat_x: float, boat_y: float, boat_heading: float) -> bool:
        """Check if boat has passed through a gate."""
        gate_vec = gate_midpoint - np.array([boat_x, boat_y])
        heading_vec = np.array([np.cos(boat_heading), np.sin(boat_heading)])
        
        distance = np.linalg.norm(gate_vec)
        dot_product = np.dot(gate_vec, heading_vec)
        
        # Passed if very close OR if gate is behind boat
        return distance < 2 * self.waypoint_pass_tol or dot_product < 0
    
    def filter_used_buoys(self, red_dict: dict, green_dict: dict) -> Tuple[dict, dict]:
        """Remove already-used buoys from dictionaries."""
        filtered_red = {k: v for k, v in red_dict.items() 
                       if k not in self.used_red_buoys}
        filtered_green = {k: v for k, v in green_dict.items() 
                         if k not in self.used_green_buoys}
        return filtered_red, filtered_green
    
    @property
    def all_gates_passed(self) -> bool:
        """Check if all gates have been passed."""
        return self.current_idx >= len(self.gates) and self.gates
    
    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return {
            'total_gates': len(self.gates),
            'current_idx': self.current_idx,
            'passed_gates': len(self.passed_gate_ids),
            'used_red': len(self.used_red_buoys),
            'used_green': len(self.used_green_buoys)
        }