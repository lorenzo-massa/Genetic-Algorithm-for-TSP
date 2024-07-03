# Genetic Algorithm for the Traveling Salesman Problem

## Overview

This project implements a genetic algorithm to solve the Traveling Salesman Problem (TSP), a classic optimization problem in computer science and operations research. The Traveling Salesman Problem involves finding the shortest possible route that visits a given set of cities and returns to the origin city. My approach uses a genetic algorithm with various operators such as selection, crossover, mutation, and local search methods to evolve a population of solutions towards an optimal or near-optimal solution.

## Features

- **Initialization**: Generates initial population using greedy algorithms and random permutations.
- **Local Search**: Employs the 2-opt algorithm to refine solutions.
- **Selection Methods**: Implements roulette wheel selection and elitism to select parents for reproduction.
- **Crossover Operators**: Uses Uniform Order Crossover (UXO) and Scramble Crossover (SCX) for generating offspring.
- **Mutation Operators**: Includes swap, inversion, scramble, and thors mutation methods.
- - **Parallelization**: Utilizes Python's `concurrent.futures` module for parallel execution of genetic operations across multiple populations.
- **Visualization**: Plots the evolution of the objective function values over generations.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project requires:
- Numpy
- Matplotlib for visualization
- numba for performance optimization

### Installation

Clone the repository and navigate to the project directory. Install necessary Python packages using pip:

```bash
pip install numpy matplotlib numba
```

### Usage

Run the solver with a specific TSP instance file:

```python
python main.py
```

Specify the TSP instance file name in the `main.py` script under `file_paths`.

## Configuration

The algorithm parameters such as population size, number of generations, mutation rate, etc., can be adjusted within the `Solver` class constructor in `main.py`.

## Contributing

Contributions are welcome. Please feel free to submit pull requests for improvements or new features.

## License

MIT License