import random
from models.solution import Solution


# default 50 ants - 200 iterations

# e so many books - 50 ants - 200 iterations - 4 runs,  50 ants - 10000 iterations - 1 run
# c - 25 ants - 50 iterations - 4 runs,  50 ants - 100 iterations - 1 run
# d - took to long
# f - 50 ants - 1000 iterations - 4 runs,  50 ants - 10000 iterations - 1 run
# B2000k_L300_D3.6k.txt - 25 ants - 50 iterations - 5 runs

class AntColonyOptimizer:
    """
    Ant Colony Optimization for library scheduling problem.
    Integrates with existing InstanceData, Library, and Solution classes.
    Each ant builds a library signup order; a solution is constructed from that order.
    Pheromone trails and heuristic desirability (eta) guide the search.
    """
    def __init__(self, instance_data, initial_solution: Solution = None ,num_ants=50, num_iterations=10000, alpha=1.0, beta=2.0, evaporation_rate=0.1, Q=1.0):
        """
        instance_data: InstanceData object containing libraries, total_days, book_scores, etc.
        initial_solution: Solution object - if provided will be used for initial path
        num_ants: number of ants to simulate per iteration.
        num_iterations: total ACO iterations.
        alpha: weight of pheromone importance.
        beta: weight of heuristic importance.
        evaporation_rate: pheromone evaporation coefficient.
        Q: pheromone deposit factor (scaling).
        """
        self.instance = instance_data
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation_rate
        self.Q = Q
        self.num_libraries = len(self.instance.libs)
        self.total_days = self.instance.num_days
        # Initialize heuristics (eta) for each library
        self.heuristic = [0.0] * self.num_libraries
        self.sorted_books = [[] for _ in range(self.num_libraries)]
        self._prepare_heuristics()
        # Compute a baseline greedy solution for initial pheromone bias or use the initial provided solution
        if initial_solution is None:
            self.greedy_path, self.greedy_score = self._greedy_solution()
        else:
            self.greedy_path   = initial_solution.signed_libraries
            self.greedy_score  = initial_solution.fitness_score
        # Initialize pheromone trails matrix and start pheromones
        self._initialize_pheromone()

    def _prepare_heuristics(self):
        """
        Precompute static heuristic value (eta) for each library based on book scores, signup time, and capacity.
        """
        scores = self.instance.scores
        for i, lib in enumerate(self.instance.libs):
            self.sorted_books[i] = sorted(lib.books, key=lambda b: scores[b.id], reverse=True)
            days_left = max(self.total_days - lib.signup_days, 0)
            max_books_scannable = days_left * lib.books_per_day
            if max_books_scannable > 0:
                top_books = self.sorted_books[i][:min(len(self.sorted_books[i]), max_books_scannable)]
                potential_score = sum(scores[b.id] for b in top_books)
                self.heuristic[i] = potential_score / lib.signup_days if lib.signup_days > 0 else potential_score
            else:
                self.heuristic[i] = 0.0

    def _greedy_solution(self):
        """
        Build a greedy solution ordering by descending heuristic, simulate scanning, and return its score and path.
        """
        available = list(range(self.num_libraries))
        available.sort(key=lambda i: self.heuristic[i], reverse=True)
        current_day = 0
        scanned_books = set()
        total_score = 0
        path = []
        for lib_idx in available:
            lib = self.instance.libs[lib_idx]
            if current_day + lib.signup_days >= self.total_days:
                continue
            path.append(lib_idx)
            current_day += lib.signup_days
            days_scanning = self.total_days - current_day
            book_quota = days_scanning * lib.books_per_day
            if book_quota <= 0:
                break
            count = 0
            for b in self.sorted_books[lib_idx]:
                if count >= book_quota:
                    break
                if b not in scanned_books:
                    scanned_books.add(b)
                    total_score += self.instance.scores[b.id]
                    count += 1
            if current_day >= self.total_days:
                break
        return path, total_score

    def _initialize_pheromone(self):
        """
        Initialize pheromone levels for edges between libraries and from start.
        Uses greedy(initial) solution to seed higher pheromone on that path.
        """
        if self.greedy_score > 0:
            tau_high = self.greedy_score
        else:
            tau_high = 1.0
        tau_low = tau_high * 0.1
        n = self.num_libraries
        self.pheromone = [[tau_low for _ in range(n)] for _ in range(n)]
        self.pheromone_start = [tau_low for _ in range(n)]
        if self.greedy_path:
            first = self.greedy_path[0]
            self.pheromone_start[first] = tau_high
            for idx in range(1, len(self.greedy_path)):
                u = self.greedy_path[idx-1]
                v = self.greedy_path[idx]
                self.pheromone[u][v] = tau_high

    def run(self):
        """
        Execute the ACO algorithm, return the best found Solution.
        """
        best_score = 0
        best_path = []
        stagnation_counter = 0
        for _ in range(self.num_iterations):
            all_paths = []
            all_scores = []
            for _ in range(self.num_ants):
                path, score, _ = self._construct_ant_solution()
                all_paths.append(path)
                all_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_path = path
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            self._update_pheromones(all_paths, all_scores, best_path, best_score)
            self.alpha += (1.0 / self.num_iterations)
            self.beta = max(0.1, self.beta - (1.0 / self.num_iterations))
            if stagnation_counter > self.num_iterations * 0.5:
                break
          
        return self._rebuild_solution(best_path)

    def _rebuild_solution(self, order: list[int]) -> Solution:
        """
        Given a signup order of library ids, rebuild the full Solution by greedily
        picking the highest-score unscanned books for each library in sequence.
        """
        curr_time = 0
        scanned = set()
        scanned_per_lib = {}
        signed = []
        for lib_id in order:
            lib = next(l for l in self.instance.libs if l.id == lib_id)
            if curr_time + lib.signup_days >= self.instance.num_days:
                break
            curr_time += lib.signup_days
            time_left = self.instance.num_days - curr_time
            capacity = time_left * lib.books_per_day
            # pick top remaining by score
            available = sorted(
                (b.id for b in lib.books if b.id not in scanned),
                key=lambda b: -self.instance.scores[b]
            )[:capacity]
            if not available:
                continue
            signed.append(lib_id)
            scanned_per_lib[lib_id] = available
            scanned.update(available)
        sol = Solution(signed, [], scanned_per_lib, scanned)
        sol.calculate_fitness_score(self.instance.scores)
        return sol
    
    def _construct_ant_solution(self):
        """
        Construct a single ant's solution (library order) probabilistically.
        Return the path (list of library indices), its score, and scanned books set.
        """
        available = set(range(self.num_libraries))
        path = []
        current_day = 0
        scanned_books = set()
        total_score = 0
        prev = None
        while available:
            feasible = []
            for j in available:
                lib = self.instance.libs[j]
                if current_day + lib.signup_days < self.total_days:
                    feasible.append(j)
            if not feasible:
                break
            weights = []
            for j in feasible:
                if prev is None:
                    tau = self.pheromone_start[j]
                else:
                    tau = self.pheromone[prev][j]
                eta = self.heuristic[j]
                weights.append((tau ** self.alpha) * (eta ** self.beta))
            total = sum(weights)
            if total <= 0:
                break
            pick = random.random() * total
            cum = 0.0
            selected = feasible[0]
            for idx, j in enumerate(feasible):
                cum += weights[idx]
                if pick <= cum:
                    selected = j
                    break
            available.remove(selected)
            path.append(selected)
            lib = self.instance.libs[selected]
            current_day += lib.signup_days
            days_scanning = self.total_days - current_day
            if days_scanning <= 0:
                break
            book_quota = days_scanning * lib.books_per_day
            count = 0
            for b in self.sorted_books[selected]:
                if count >= book_quota:
                    break
                if b not in scanned_books:
                    scanned_books.add(b)
                    total_score += self.instance.scores[b.id]
                    count += 1
            prev = selected
            if current_day >= self.total_days:
                break
        return path, total_score, scanned_books

    def _update_pheromones(self, all_paths, all_scores, best_path, best_score):
        """
        Evaporate pheromones and deposit new pheromones based on paths and scores.
        Uses best path to intensify search.
        """
        n = self.num_libraries
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] *= (1 - self.rho)
            self.pheromone_start[i] *= (1 - self.rho)
        for path, score in zip(all_paths, all_scores):
            deposit = self.Q * score
            if not path:
                continue
            first = path[0]
            self.pheromone_start[first] += deposit
            for idx in range(1, len(path)):
                u = path[idx-1]
                v = path[idx]
                self.pheromone[u][v] += deposit
        if best_path:
            elite_deposit = self.Q * best_score
            first = best_path[0]
            self.pheromone_start[first] += elite_deposit
            for idx in range(1, len(best_path)):
                u = best_path[idx-1]
                v = best_path[idx]
                self.pheromone[u][v] += elite_deposit