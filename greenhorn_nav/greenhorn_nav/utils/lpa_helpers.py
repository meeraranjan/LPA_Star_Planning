#lpa_helpers.py
import heapq
import numpy as np

class LPAStar:
    """LPA* / D* Lite path planning helper.

    Implements incremental shortest path planning on a grid.
    - rhs[goal] = 0 is the source of backward propagation.
    - compute_shortest_path() runs until start node is locally consistent.
    - g[s] approximates shortest-cost-from-s to goal.
    """

    def __init__(self, start, goal, grid, heuristic='euclidean'):
        """Initialize the LPA* planner.

        Args:
            start (tuple): Start node indices (i, j).
            goal (tuple): Goal node indices (i, j).
            grid (np.ndarray): Occupancy grid (0=free, >0=obstacle).
            heuristic (str, optional): 'euclidean' or 'manhattan'. Defaults to 'euclidean'.
        """
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.grid = grid
        self.rhs = {}
        self.g = {}
        self.U = []    # heap of (key_tuple, node)
        self.U_set = set()  # Track nodes in U for efficient lookup
        self.km = 0.0
        self.heuristic = self.euclidean if heuristic == 'euclidean' else self.manhattan

    def manhattan(self, a, b):
        """Compute Manhattan distance between two nodes."""
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def euclidean(self, a, b):
        """Compute Euclidean distance between two nodes."""
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def initialize(self):
        """Initialize or reset internal containers for a fresh planning run."""
        self.rhs.clear()
        self.g.clear()
        self.U.clear()
        self.U_set.clear()
        # For LPA*/D*Lite we set rhs[goal] = 0 (goal is the source)
        self.rhs[self.goal] = 0.0
        self.g[self.goal] = np.inf
        key = self.calculate_key(self.goal)
        heapq.heappush(self.U, (key, self.goal))
        self.U_set.add(self.goal)

    def calculate_key(self, s):
        """Calculate the LPA* key for a node."""
        g_rhs = min(self.g.get(s, np.inf), self.rhs.get(s, np.inf))
        k1 = g_rhs + self.heuristic(s, self.start) + self.km
        k2 = g_rhs
        return (k1, k2)

    def neighbors(self, s):
        """Get valid neighboring nodes for a given node."""
        x, y = s
        moves = [
            (1,0), (-1,0), (0,1), (0,-1),
            (1,1), (1,-1), (-1,1), (-1,-1)
        ]
        nbrs = []
        max_x = self.grid.shape[0]
        max_y = self.grid.shape[1]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y:
                nbrs.append((nx, ny))
        return nbrs

    def cost(self, s1, s2):
        """Compute the cost to move from s1 to s2."""
        # Check if target cell is an obstacle
        try:
            if self.grid[s2] > 0:
                return np.inf
        except Exception:
            return np.inf
        
        dx = abs(s1[0] - s2[0])
        dy = abs(s1[1] - s2[1])
        if dx + dy == 2:
            return np.sqrt(2.0)
        return 1.0

    def update_vertex(self, s):
        """Update the rhs value of a node and adjust its presence in the priority queue."""
        # Update rhs value (except for goal)
        if s != self.goal:
            nbrs = self.neighbors(s)
            best = np.inf
            for sp in nbrs:
                val = self.cost(s, sp) + self.g.get(sp, np.inf)
                if val < best:
                    best = val
            self.rhs[s] = best

        # Remove s from U if present (using set for efficient lookup)
        if s in self.U_set:
            # Remove from heap (rebuild is easier than trying to find and remove)
            self.U = [(k, v) for (k, v) in self.U if v != s]
            heapq.heapify(self.U)
            self.U_set.remove(s)

        # If locally inconsistent, add to U
        g_s = self.g.get(s, np.inf)
        rhs_s = self.rhs.get(s, np.inf)
        if g_s != rhs_s:
            key = self.calculate_key(s)
            heapq.heappush(self.U, (key, s))
            self.U_set.add(s)

    def compute_shortest_path(self):
        """Compute the shortest path from start to goal."""
        iterations = 0
        max_iterations = 100000
        
        while self.U and iterations < max_iterations:
            iterations += 1
            
            # Get top key and node
            top_key, top_node = self.U[0]
            
            # Check termination condition
            start_key = self.calculate_key(self.start)
            g_start = self.g.get(self.start, np.inf)
            rhs_start = self.rhs.get(self.start, np.inf)
            
            # Terminate if start is consistent and has better key than top
            if (top_key >= start_key) and (g_start == rhs_start):
                break

            # Pop node with smallest key
            k_old, u = heapq.heappop(self.U)
            self.U_set.discard(u)
            
            # Recalculate key
            k_new = self.calculate_key(u)
            
            # If key changed, reinsert with new key
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
                self.U_set.add(u)
            else:
                g_u = self.g.get(u, np.inf)
                rhs_u = self.rhs.get(u, np.inf)
                
                if g_u > rhs_u:
                    # Overconsistent: make consistent
                    self.g[u] = rhs_u
                    # Update all predecessors
                    for s in self.neighbors(u):
                        self.update_vertex(s)
                else:
                    # Underconsistent: raise g to infinity
                    self.g[u] = np.inf
                    # Update u and all predecessors
                    self.update_vertex(u)
                    for s in self.neighbors(u):
                        self.update_vertex(s)
        
        if iterations >= max_iterations:
            print(f"WARNING: compute_shortest_path reached max iterations ({max_iterations})")
        
        # print(f"compute_shortest_path completed in {iterations} iterations")
        # print(f"Start g={self.g.get(self.start, np.inf):.2f}, rhs={self.rhs.get(self.start, np.inf):.2f}")