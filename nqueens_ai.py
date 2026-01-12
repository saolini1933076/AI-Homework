import heapq
import time
from ortools.sat.python import cp_model
import csv

# ==========================
# N-Queens Problem Definition
# ==========================
class NQueensProblem:
    def __init__(self, n):
        self.n = n

    def initial_state(self):
        return tuple()

    def is_goal(self, state):
        return len(state) == self.n and self.is_valid(state)

    def actions(self, state):
        row = len(state)
        for col in range(self.n):
            yield col

    def result(self, state, action):
        return state + (action,)

    def cost(self, state, action):
        return 1

    def is_valid(self, state):
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j]:
                    return False
                if abs(state[i] - state[j]) == abs(i - j):
                    return False
        return True

# ==========================
# Heuristic
# ==========================
def conflict_heuristic(state):
    conflicts = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] == state[j]:
                conflicts += 1
            if abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return conflicts

# ==========================
# A* Implementation
# ==========================
class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def astar_search(problem, heuristic):
    start_time = time.time()
    start_state = problem.initial_state()
    start_node = Node(start_state, g=0, h=heuristic(start_state))

    frontier = []
    heapq.heappush(frontier, start_node)
    explored = set()

    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = heapq.heappop(frontier)

        if problem.is_goal(node.state):
            return {
                "solution": reconstruct_path(node),
                "expanded": expanded,
                "generated": generated,
                "max_frontier": max_frontier,
                "time": time.time() - start_time
            }

        explored.add(node.state)
        expanded += 1

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if not problem.is_valid(child_state):
                continue
            if child_state in explored:
                continue

            child_node = Node(
                child_state,
                parent=node,
                g=node.g + problem.cost(node.state, action),
                h=heuristic(child_state)
            )
            heapq.heappush(frontier, child_node)
            generated += 1

    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))

# ==========================
# CSP Solver
# ==========================
def solve_nqueens_csp(n):
    start_time = time.time()
    model = cp_model.CpModel()
    queens = [model.NewIntVar(0, n - 1, f"Q{i}") for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            model.Add(queens[i] != queens[j])
            model.AddAbsEquality(model.NewIntVar(0, n, ""), queens[i] - queens[j])
            model.Add(abs(i - j) != abs(queens[i] - queens[j]))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = [solver.Value(q) for q in queens]
        return {
            "solution": solution,
            "time": time.time() - start_time,
            "branches": solver.NumBranches(),
            "conflicts": solver.NumConflicts()
        }
    return None

# ==========================
# Experiments
# ==========================
def run_experiments():
    ns = [4, 6, 8, 10, 12]

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Algorithm", "N", "Time",
            "Expanded", "Generated", "MaxFrontier",
            "Branches", "Conflicts"
        ])

        for n in ns:
            print(f"Running A* for n={n}")
            problem = NQueensProblem(n)
            result = astar_search(problem, conflict_heuristic)
            writer.writerow([
                "A*", n, result["time"], result["expanded"],
                result["generated"], result["max_frontier"], "", ""
            ])

            print(f"Running CSP for n={n}")
            csp_result = solve_nqueens_csp(n)
            writer.writerow([
                "CSP", n, csp_result["time"], "", "", "",
                csp_result["branches"], csp_result["conflicts"]
            ])

    print("Experiments complete! Results saved to results.csv")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    run_experiments()
