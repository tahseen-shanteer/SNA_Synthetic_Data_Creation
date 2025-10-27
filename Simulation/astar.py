import numpy as np
import heapq

class AStar:
    def __init__(self, grid):
        """
        Initialize A* algorithm with a 2D numpy array grid.
        Values: 0 = walkable, 1 = obstacle
        """
        self.grid = grid
        self.rows, self.cols = grid.shape

    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        """Get valid neighboring cells (4-direction movement)"""
        row, col = node
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.grid[new_row, new_col] == 0):
                neighbors.append((new_row, new_col))
        return neighbors

    def find_path(self, start, goal):
        """
        Find path from start to goal using A* algorithm.
        Returns list of coordinates representing the path, or None if no path exists.
        """
        if start == goal:
            return [start]

        open_set = []
        heapq.heappush(open_set, (0, start))
        open_set_nodes = {start}

        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        came_from = {}

        closed_set = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_nodes.discard(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    if neighbor not in open_set_nodes:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_nodes.add(neighbor)
        return None