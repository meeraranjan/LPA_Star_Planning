"""
Support components for the LPA* planner node.

This module contains configuration, state management, and helper classes
that handle specific aspects of the planning system.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field

from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

from greenhorn_nav.utils.grid_processing import inflate_obstacles, detect_changed_cells
from greenhorn_nav.utils.lpa_helpers import LPAStar


# Type aliases
MapIndex = Tuple[int, int]
WorldXY = Tuple[float, float]


@dataclass
class PlannerConfig:
    """Configuration parameters for the LPA planner."""
    
    # Topic names
    map_topic: str = '/map'
    gate_midpoints_topic: str = '/gate_midpoints'
    boat_pose_topic: str = '/boat_pose'
    local_path_topic: str = '/local_path'
    frame_id: str = 'map'
    
    # Planning parameters
    inflation_radius: float = 0.1  # meters
    cycle_hz: float = 2.0
    planner_heuristic: str = 'euclidean'
    max_extraction_iters: int = 20000
    
    # Gate navigation
    planning_horizon_distance: float = 15.0  # meters
    min_gates_ahead: int = 1
    max_gates_ahead: int = 4
    blocked_gate_retry_limit: int = 5
    allow_gate_skipping: bool = True
    gate_passed_proximity_threshold: float = 1.5  # meters
    
    # Smoothing
    smoothing_enabled: bool = True
    smoothing_factor: float = 0.5
    smoothing_min_points: int = 10
    smoothing_max_points: int = 100
    smoothing_multiplier: int = 2
    
    # Grid processing
    cell_occupied_threshold: int = 50
    qos_depth: int = 10


@dataclass
class PlannerState:
    """Current state of the planner."""
    
    map_msg: Optional[OccupancyGrid] = None
    inflated: Optional[np.ndarray] = None
    prev_inflated: Optional[np.ndarray] = None
    gate_midpoints: List[WorldXY] = field(default_factory=list)
    boat_pose: Optional[WorldXY] = None
    lpa_instances: List[Optional[LPAStar]] = field(default_factory=list)
    last_start: Optional[MapIndex] = None
    current_gate_idx: int = 0
    gate_blocked_count: Dict[int, int] = field(default_factory=dict)
    skipped_gates: Set[int] = field(default_factory=set)
    passed_gates: Set[int] = field(default_factory=set)


class MapCoordinateTransformer:
    """Handles coordinate transformations between world and map frames."""
    
    def __init__(self, node: Node):
        self.node = node
        self._map_msg: Optional[OccupancyGrid] = None
        self._inflated: Optional[np.ndarray] = None
    
    def update_map(self, map_msg: OccupancyGrid, inflated: np.ndarray):
        """Update the stored map information."""
        self._map_msg = map_msg
        self._inflated = inflated
    
    def world_to_map(self, xy: WorldXY) -> Optional[MapIndex]:
        """Convert world coordinates (x,y) to grid indices (i,j)."""
        if self._map_msg is None or self._inflated is None:
            return None
        
        ox = self._map_msg.info.origin.position.x
        oy = self._map_msg.info.origin.position.y
        res = self._map_msg.info.resolution
        
        j = int((xy[0] - ox) / res)
        i = int((xy[1] - oy) / res)
        
        if 0 <= i < self._inflated.shape[0] and 0 <= j < self._inflated.shape[1]:
            return (i, j)
        return None
    
    def map_to_world(self, ij: MapIndex) -> WorldXY:
        """Convert grid index (i,j) to world coordinates (x,y) at cell center."""
        assert self._map_msg is not None
        
        ox = self._map_msg.info.origin.position.x
        oy = self._map_msg.info.origin.position.y
        res = self._map_msg.info.resolution
        
        x = ox + ij[1] * res + res * 0.5
        y = oy + ij[0] * res + res * 0.5
        return (x, y)


class GateNavigator:
    """Manages gate waypoint navigation and tracking."""
    
    def __init__(self, node: Node, config: PlannerConfig, transformer: MapCoordinateTransformer):
        self.node = node
        self.config = config
        self.transformer = transformer
    
    def update_passed_gates(self, state: PlannerState):
        """Mark gates as passed if the boat is behind them."""
        if not state.boat_pose or not state.gate_midpoints:
            return
        
        boat_pos = np.array(state.boat_pose)
        
        for gate_idx in range(len(state.gate_midpoints)):
            if gate_idx in state.passed_gates:
                continue
            
            if self._is_gate_passed(boat_pos, state.gate_midpoints, gate_idx):
                state.passed_gates.add(gate_idx)
                self.node.get_logger().info(f"Marked gate {gate_idx} as passed")
    
    def _is_gate_passed(self, boat_pos: np.ndarray, gates: List[WorldXY], gate_idx: int) -> bool:
        """Check if a gate has been passed by the boat."""
        gate_pos = np.array(gates[gate_idx])
        to_gate = gate_pos - boat_pos
        dist = np.linalg.norm(to_gate)
        
        if dist < self.config.gate_passed_proximity_threshold:
            # Check if we're heading toward the next gate
            if gate_idx < len(gates) - 1:
                next_gate_pos = np.array(gates[gate_idx + 1])
                to_next = next_gate_pos - boat_pos
                dist_to_next = np.linalg.norm(to_next)
                
                if dist_to_next < dist:
                    return True
            else:
                # Last gate - mark as passed if very close
                if dist < 0.3:
                    return True
        
        return False
    
    def advance_current_gate(self, state: PlannerState):
        """Advance to the next gate if current gate is passed."""
        while (state.current_gate_idx in state.passed_gates and 
               state.current_gate_idx < len(state.gate_midpoints) - 1):
            self.node.get_logger().info(
                f"Gate {state.current_gate_idx} is behind us - advancing"
            )
            state.current_gate_idx += 1
    
    def compute_planning_horizon(self, state: PlannerState, start: MapIndex) -> int:
        """Compute how many gates ahead to plan based on distance."""
        if not state.gate_midpoints:
            return 0
        
        boat_world = self.transformer.map_to_world(start)
        boat_pos = np.array(boat_world)
        
        end_gate_idx = state.current_gate_idx
        cumulative_distance = 0.0
        
        for gate_idx in range(state.current_gate_idx, len(state.gate_midpoints)):
            gate_pos = np.array(state.gate_midpoints[gate_idx])
            
            if gate_idx == state.current_gate_idx:
                dist_to_gate = np.linalg.norm(gate_pos - boat_pos)
            else:
                prev_gate_pos = np.array(state.gate_midpoints[gate_idx - 1])
                dist_to_gate = np.linalg.norm(gate_pos - prev_gate_pos)
            
            cumulative_distance += dist_to_gate
            
            if cumulative_distance > self.config.planning_horizon_distance:
                break
            
            end_gate_idx = gate_idx
            
            if (gate_idx - state.current_gate_idx + 1) >= self.config.max_gates_ahead:
                break
        
        # Ensure minimum gates ahead
        min_end_idx = min(
            state.current_gate_idx + self.config.min_gates_ahead - 1,
            len(state.gate_midpoints) - 1
        )
        end_gate_idx = max(end_gate_idx, min_end_idx)
        
        self.node.get_logger().debug(
            f"Planning horizon: {cumulative_distance:.2f}m covers gates "
            f"{state.current_gate_idx} to {end_gate_idx}"
        )
        
        return end_gate_idx


class SegmentPlanner:
    """Plans individual path segments between waypoints using LPA*."""
    
    def __init__(self, node: Node, config: PlannerConfig, transformer: MapCoordinateTransformer):
        self.node = node
        self.config = config
        self.transformer = transformer
    
    def plan_segment(self, start: MapIndex, goal: MapIndex, gate_idx: int, 
                     state: PlannerState) -> List[WorldXY]:
        """Plan a path segment from start to a specific gate."""
        # Ensure we have enough LPA instances
        while len(state.lpa_instances) <= gate_idx:
            state.lpa_instances.append(None)
        
        lpa = state.lpa_instances[gate_idx]
        
        # Initialize or recreate LPA* if needed
        if lpa is None or lpa.goal != goal:
            if lpa is not None and lpa.goal != goal:
                self.node.get_logger().info(
                    f"Gate {gate_idx} goal changed -> recreating planner"
                )
            if not self._create_lpa(start, goal, gate_idx, state):
                return []
            lpa = state.lpa_instances[gate_idx]
        
        # Ensure grid reference is current
        lpa.grid = state.inflated
        
        # Handle moving start
        if start != lpa.start:
            if gate_idx == state.current_gate_idx:
                prev_start = lpa.start
                delta_h = lpa.heuristic(prev_start, start)
                lpa.km += delta_h
                self.node.get_logger().debug(
                    f"Gate {gate_idx}: Start moved, km += {delta_h:.3f}"
                )
            lpa.start = start
        
        # Run LPA*
        try:
            self.node.get_logger().debug(
                f"Running compute_shortest_path for gate {gate_idx}"
            )
            lpa.compute_shortest_path()
        except Exception as ex:
            self.node.get_logger().error(
                f"compute_shortest_path failed for gate {gate_idx}: {ex}"
            )
            return []
        
        # Check reachability
        start_g = lpa.g.get(start, np.inf)
        if start_g == np.inf:
            self.node.get_logger().warn(
                f"No path to gate {gate_idx}: start has infinite g-value"
            )
            return []
        
        self.node.get_logger().debug(
            f"Gate {gate_idx}: Path found with start g-value: {start_g:.2f}"
        )
        
        return self._extract_path(start, goal, lpa)
    
    def _create_lpa(self, start: MapIndex, goal: MapIndex, gate_idx: int,
                    state: PlannerState) -> bool:
        """Create and initialize LPA* instance for a specific gate."""
        try:
            lpa = LPAStar(
                start=start,
                goal=goal,
                grid=state.inflated,
                heuristic=self.config.planner_heuristic
            )
            lpa.km = 0.0
            lpa.initialize()
            state.lpa_instances[gate_idx] = lpa
            state.last_start = start
            return True
        except Exception as ex:
            self.node.get_logger().error(
                f"Failed to create LPA* for gate {gate_idx}: {ex}"
            )
            state.lpa_instances[gate_idx] = None
            return False
    
    def _extract_path(self, start: MapIndex, goal: MapIndex, lpa: LPAStar) -> List[WorldXY]:
        """Extract a path from start to goal using LPA*'s g-values."""
        if lpa.g.get(start, np.inf) == np.inf:
            return []
        
        path: List[MapIndex] = [start]
        cur = start
        
        for it in range(self.config.max_extraction_iters):
            if cur == goal:
                break
            
            nbrs = lpa.neighbors(cur)
            best = None
            best_val = np.inf
            
            for n in nbrs:
                g_n = lpa.g.get(n, np.inf)
                if g_n == np.inf:
                    continue
                
                val = lpa.cost(cur, n) + g_n
                if val < best_val:
                    best_val = val
                    best = n
            
            if best is None:
                break
            
            path.append(best)
            cur = best
        else:
            self.node.get_logger().warning("Reached max extraction iterations")
        
        # Convert to world coordinates
        return [self.transformer.map_to_world(p) for p in path]


class MapProcessor:
    """Processes occupancy grid maps and handles obstacle inflation."""
    
    def __init__(self, node: Node, config: PlannerConfig):
        self.node = node
        self.config = config
    
    def process_map(self, msg: OccupancyGrid, state: PlannerState) -> bool:
        """Process incoming occupancy grid map."""
        try:
            state.map_msg = msg
            width = msg.info.width
            height = msg.info.height
            res = msg.info.resolution
            
            arr = np.array(msg.data, dtype=np.int8).reshape((height, width))
            binmap = (arr > self.config.cell_occupied_threshold).astype(np.uint8)
            
            if state.inflated is not None:
                state.prev_inflated = state.inflated.copy()
            
            new_inflated = inflate_obstacles(
                binmap,
                inflation_radius=self.config.inflation_radius,
                resolution=res
            )
            
            # Detect and handle changes
            if state.prev_inflated is not None:
                changed = detect_changed_cells(state.prev_inflated, new_inflated)
                if changed:
                    state.inflated = new_inflated
                    self._update_lpa_instances(changed, state)
                else:
                    state.inflated = new_inflated
            else:
                state.inflated = new_inflated
            
            return True
            
        except Exception as ex:
            self.node.get_logger().error(f"map processing failed: {ex}")
            return False
    
    def _update_lpa_instances(self, changed_cells: Set[MapIndex], state: PlannerState):
        """Update all LPA* instances with changed cells."""
        for gate_idx, lpa in enumerate(state.lpa_instances):
            if lpa is not None:
                lpa.grid = state.inflated
                for cell in changed_cells:
                    try:
                        lpa.update_vertex(cell)
                        for nbr in lpa.neighbors(cell):
                            lpa.update_vertex(nbr)
                    except Exception as ex:
                        self.node.get_logger().error(
                            f"update_vertex failed for gate {gate_idx}, "
                            f"cell {cell}: {ex}"
                        )