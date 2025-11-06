#lpa_helpers.py
import heapq
import numpy as np

class LPAStar:
    """LPA* / D* Lite path planning helper.

    Implements incremental shortest path planning on a grid.
    - rhs[goal] = 0 is the source of backward propagation.
    - compute_shortest_path() runs until start node is locally consistent.
    - g[s] approximates shortest-cost-from-s to goal.
    
    Attributes:
        start (tuple): Start node (i, j) in grid coordinates.
        goal (tuple): Goal node (i, j) in grid coordinates.
        grid (np.ndarray): Occupancy grid (0=free, >0=obstacle).
        rhs (dict): One-step lookahead values for each node.
        g (dict): Current cost-to-go values for each node.
        U (list): Priority queue (heap) of nodes to process.
        km (float): LPA* accumulated heuristic adjustment for start moves.
        heuristic (Callable): Heuristic function (euclidean or manhattan).
    """

    def __init__(self, start, goal, grid, heuristic='euclidean'):
        """Initialize the LPA* planner.

        Args:
            start (tuple): Start node indices (i, j).
            goal (tuple): Goal node indices (i, j).
            grid (np.ndarray): Occupancy grid (0=free, >0=obstacle).
            heuristic (str, optional): 'euclidean' or 'manhattan'. Defaults to 'euclidean'.
        """
        # store as tuples
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.grid = grid
        self.rhs = {}
        self.g = {}
        self.U = []    # heap of (key_tuple, node)
        self.km = 0.0
        self.heuristic = self.euclidean if heuristic == 'euclidean' else self.manhattan

    def manhattan(self, a, b):
        """Compute Manhattan distance between two nodes.

        Args:
            a (tuple): Node (i, j).
            b (tuple): Node (i, j).

        Returns:
            float: Manhattan distance.
        """
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def euclidean(self, a, b):
        """Compute Euclidean distance between two nodes.

        Args:
            a (tuple): Node (i, j).
            b (tuple): Node (i, j).

        Returns:
            float: Euclidean distance.
        """
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def initialize(self):
        """Initialize or reset internal containers for a fresh planning run.

        Sets rhs[goal] = 0 (goal as source) and pushes goal into the priority queue.
        """
        # clear internal containers for a fresh start
        self.rhs.clear()
        self.g.clear()
        self.U.clear()
        # For LPA*/D*Lite we set rhs[goal] = 0 (goal is the source)
        self.rhs[self.goal] = 0.0
        self.g[self.goal] = np.inf
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, s):
        """Calculate the LPA* key for a node.

        Key is a tuple (k1, k2):
            k1 = min(g, rhs) + h(s, start) + km
            k2 = min(g, rhs)

        Args:
            s (tuple): Node (i, j).

        Returns:
            tuple: Node key (k1, k2) for the priority queue.
        """
        # key is a tuple (k1, k2) where k1 = min(g,rhs) + h(s, start) + km, k2 = min(g,rhs)
        g_rhs = min(self.g.get(s, np.inf), self.rhs.get(s, np.inf))
        k1 = g_rhs + self.heuristic(s, self.start) + self.km
        k2 = g_rhs
        return (k1, k2)

    def neighbors(self, s):
        """Get valid neighboring nodes for a given node.

        Includes 8-connected neighbors (diagonal moves allowed).

        Args:
            s (tuple): Node (i, j).

        Returns:
            list: List of neighbor nodes (i, j) within grid bounds.
        """
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
        """Compute the cost to move from s1 to s2.

        Args:
            s1 (tuple): Node (i, j).
            s2 (tuple): Node (i, j).

        Returns:
            float: Cost (1.0 for cardinal, sqrt(2) for diagonal, inf if obstacle).
        """
        # If either cell is an obstacle, treat as infinite cost
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
        """Update the rhs value of a node and adjust its presence in the priority queue.

        Args:
            s (tuple): Node (i, j) to update.
        """
        if s != self.goal:
            nbrs = self.neighbors(s)
            best = np.inf
            best_nbr = None
            for sp in nbrs:
                val = self.g.get(sp, np.inf) + self.cost(s, sp)
                if val < best:
                    best = val
                    best_nbr = sp
            old_rhs = self.rhs.get(s, np.inf)
            self.rhs[s] = best
            if old_rhs != best:
                print(f"update_vertex: {s} rhs {old_rhs:.2f} -> {best:.2f}, best neighbor={best_nbr}")

        # remove s from U if present
        present = False
        for k, v in self.U:
            if v == s:
                present = True
                break
        if present:
            self.U = [(k,v) for (k,v) in self.U if v != s]
            heapq.heapify(self.U)

        # if g[s] != rhs[s], push/update in U
        g_s = self.g.get(s, np.inf)
        rhs_s = self.rhs.get(s, np.inf)
        if g_s != rhs_s:
            heapq.heappush(self.U, (self.calculate_key(s), s))
            print(f"update_vertex: pushed {s} to U, g={g_s:.2f}, rhs={rhs_s:.2f}, key={self.calculate_key(s)}")

    def compute_shortest_path(self):
        """Compute the shortest path from start to goal.

        Runs until the start node is locally consistent (g[start] = rhs[start]).
        Updates g-values and propagates changes through neighbors.
        """
        print(f"compute_shortest_path: start g={self.g.get(self.start, np.inf)}, rhs={self.rhs.get(self.start, np.inf)}")

        # run until start is consistent
        while self.U:
            top_key, top_node = self.U[0]
            # compare first key component to start's key first component
            start_key = self.calculate_key(self.start)
            if top_key[0] >= start_key[0] and self.rhs.get(self.start, np.inf) == self.g.get(self.start, np.inf):
                print("compute_shortest_path: start node consistent, done")
                break

            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            else:
                old_g = self.g.get(u, np.inf)
                old_rhs = self.rhs.get(u, np.inf)
                if old_g > old_rhs:
                    self.g[u] = old_rhs
                    print(f"compute_shortest_path: g[{u}] {old_g:.2f} -> {old_rhs:.2f}")
                    for s in self.neighbors(u):
                        # print(f"Obstacle at {s}, grid value={self.grid[s]}")

                        self.update_vertex(s)
                else:
                    self.g[u] = np.inf
                    print(f"compute_shortest_path: g[{u}] set to inf")
                    for s in self.neighbors(u) + [u]:
                        # print(f"Obstacle at {s}, grid value={self.grid[s]}")

                        self.update_vertex(s)
