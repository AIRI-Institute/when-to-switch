from heapq import heappop, heappush
INF = 1000000000


class Node:
    def __init__(self, coord: (int, int) = (INF, INF), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or \
               (self.f == other.f and (self.g < other.g or
                                       (self.g == other.g and (self.i < other.i or
                                                               (self.i == other.i and self.j < other.j)))))


class AStar:
    def __init__(self, max_steps: int = INF):
        self.start = None
        self.goal = None
        self.max_steps = max_steps
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.best_node = None
        self.desired_position = None
        self.bad_actions = set()

    def h(self, node: [int, int]):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def get_neighbours(self, u: [int, int]):
        neighbors = []
        for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (u[0] + d[0], u[1] + d[1]) not in self.obstacles:
                neighbors.append((u[0] + d[0], u[1] + d[1]))
        return neighbors

    def compute_shortest_path(self):
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            if self.best_node.h > u.h:
                self.best_node = u
            steps += 1
            for n in self.get_neighbours((u.i, u.j)):
                if n not in self.CLOSED and n not in self.other_agents:
                    heappush(self.OPEN, Node(n, u.g + 1, self.h(n)))
                    self.CLOSED[n] = (u.i, u.j)

    def get_next_node(self, use_best_node: bool):
        next_node = None
        if self.goal in self.CLOSED:
            next_node = self.goal
        elif use_best_node:
            next_node = (self.best_node.i, self.best_node.j)
        if next_node is not None and next_node != self.start:
            while self.CLOSED[next_node] != self.start:
                next_node = self.CLOSED[next_node]
            self.desired_position = next_node
            return self.start, next_node
        else:
            self.desired_position = None
            return None

    def update_obstacles(self, obs, other_agents, n):
        for o in obs:
            self.obstacles.add((n[0] + o[0], n[1] + o[1]))
        self.other_agents.clear()
        for a in other_agents:
            self.other_agents.add((n[0] + a[0], n[1] + a[1]))

    def reset(self):
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start, 0, self.h(self.start)))
        self.best_node = Node(self.start, 0, self.h(self.start))

    def update_path(self, start, goal):
        if self.desired_position and self.desired_position != start:
            self.bad_actions.add(self.desired_position)
            if self.start == start:
                self.other_agents = self.other_agents.union(self.bad_actions)
        else:
            self.bad_actions.clear()
        self.start = start
        self.goal = goal
        self.reset()
        self.compute_shortest_path()

    def get_path(self, use_best_node: bool):
        path = []
        next_node = None
        if self.goal in self.CLOSED:
            next_node = self.goal
        elif use_best_node:
            next_node = (self.best_node.i, self.best_node.j)
        if next_node is not None and next_node != self.start:
            while self.CLOSED[next_node] != self.start:
                path.append(next_node)
                next_node = self.CLOSED[next_node]
            path.append(next_node)
            path.append(self.start)
            self.desired_position = next_node
            path.reverse()
            return path
        else:
            self.desired_position = None
            return None