# AIN_HH_ACO

## HyperHeuristicSolver

The `HyperHeuristicSolver` is a core component of this project, designed to solve the Book Scanning problem using a hyper-heuristic approach. It incrementally builds a solution by selecting and applying a variety of low-level heuristics, adapting its strategy based on observed improvements.

### Key Features

- **Multiple Heuristics:**  
  The solver includes a diverse set of heuristics for selecting which library to add next, such as:
  - Most books
  - Lowest signup-to-score coefficient
  - Highest scanning capacity
  - Highest total book score
  - Unique books
  - Time-aware (fastest signup)
  - Hybrid ratio (score, signup, throughput)
  - Highest unscanned score
  - Density (score per signup day)
  - Throughput utilization
  - Minimal overlap

- **Adaptive Selection:**  
  Heuristics are chosen in each iteration, and the solver prefers those that yield better solutions.

- **Incremental Solution Building:**  
  Libraries are added one at a time, with careful selection of books to maximize score within the time constraints.

- **Logging:**  
  The solver logs which heuristics were used at each step, enabling post-run analysis and further tuning.

### Usage Example

To use the `HyperHeuristicSolver`, instantiate it and call `solve` with an `InstanceData` object:

```python
from models.hyper_heuristics_solver import HyperHeuristicSolver

solver = HyperHeuristicSolver(time_limit=300, iterations=10000)
solution, heuristic_log = solver.solve(instance_data)
```
-   `solution` contains the final set of libraries and books to scan.
-   `heuristic_log` provides the sequence of heuristics used.
   
### Customization

-   **Time Limit & Iterations:**\
    You can set the maximum runtime and number of iterations via the constructor.
-   **Heuristic Set:**\
    You can add or remove heuristics by modifying the `self.heuristics` list in the class.

This solver is especially useful for large and complex instances where a single heuristic may not suffice. By combining multiple strategies and adapting over time, it aims to find high-quality solutions efficiently.

## Ant Colony Optimization (ACO) Solver

The `AntColonyOptimizer` class implements Ant Colony Optimization (ACO) algorithm for refining solutions to the Book Scanning (library scheduling) problem. This solver attempt to find an optimal or near-optimal order in which to sign up libraries, maximizing the total score of scanned books within the given time constraints.

### Key Features

- **Metaheuristic Search:**  
  Uses the Ant Colony Optimization paradigm, where multiple "ants" probabilistically construct solutions based on pheromone trails and heuristic desirability.
- **Pheromone Trails:**  
  Pheromone values are updated based on the quality of solutions found, guiding future ants toward promising library orders.
- **Heuristic Guidance:**  
  Static heuristics (such as average book score per signup day) help ants make informed choices.
- **Flexible Parameters:**  
  Supports tuning of the number of ants, iterations, pheromone evaporation rate, and the influence of pheromone vs. heuristic information.
- **Solution Refinement:**  
  Can take an initial solution and attempt to improve it, or build a solution from scratch.

### Usage Example

To use the ACO optimizer, instantiate it with your problem data and an initial solution, then call `run()`:

```python

from models.aco import AntColonyOptimizer

aco = AntColonyOptimizer(
    instance_data=instance_data,
    initial_solution=initial_solution,
    num_ants=20,
    num_iterations=100,
    alpha=1.0,
    beta=2.0,
    evaporation_rate=0.1,
    Q=1.0
)
best_solution = aco.run()
```

### Parameters

-   `num_ants`: Number of ants (candidate solutions) per iteration.
-   `num_iterations`: Number of optimization iterations.
-   `alpha`: Influence of pheromone trails.
-   `beta`: Influence of heuristic information.
-   `evaporation_rate` / `rho`: Rate at which pheromone trails evaporate.
-   `pheromone_deposit` / `Q`: Amount of pheromone deposited for good solutions.
  
### Implementation Details

-   **Library Order Optimization:**\
    The algorithm focuses on finding the best order to sign up libraries, as this order greatly affects the total score.
-   **Solution Construction:**\
    Each ant builds a candidate order, and the best solutions are used to reinforce pheromone trails.
-   **Heuristic Calculation:**\
    Heuristics are based on potential book scores, signup times, and scanning capacities.

This ACO solver is suitable for large, combinatorial scheduling problems where greedy or single-heuristic approaches may get stuck in local optima. By simulating collective search and learning, it can discover high-quality solutions efficiently.