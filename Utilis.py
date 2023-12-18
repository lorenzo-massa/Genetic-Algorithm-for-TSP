import numpy as np
import random
import matplotlib as plt

def plot_graph(mean, best):
    plt.plot(mean, label='Mean fitness')
    plt.title('Mean fitness convergence')
    plt.xlabel('Iterations')
    plt.ylabel('mean fitness')

    plt.plot(best, label='Best fitness')
    plt.title('Best fitness convergence')
    plt.xlabel('Iterations')
    plt.ylabel('best fitness')
    plt.legend()
    plt.show()

def longest_common_subpath(array1, array2):
    m, n = len(array1), len(array2)
    # Initialize a matrix to store the lengths of the longest common subpaths
    dp = np.zeros((m + 1, n + 1), dtype=int)
    # Variables to store the length of the longest common subpath and its ending position
    max_length = 0
    ending_pos = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if array1[i - 1] == array2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_pos = i
            else:
                dp[i][j] = 0
    # Retrieve the longest subpath by slicing the array using the ending position and length
    longest_subpath = array1[ending_pos - max_length:ending_pos]
    return longest_subpath

def one_opt(self, population: np.ndarray, k):

    modified = 0
    if k < 1 or k > population.shape[0] - 1:
        raise ValueError("k must be between 2 and n-1")

    for i in range(population.shape[0]):
        # For each individual in the population
        best_tour = population[i, :]
        best_obj = self.objf(best_tour)
        ri_list = []

        for j in range(k):
            # Select a random index
            ri = random.randrange(1, self.adjacency_mat.shape[0]-1)
            # Check if the index is already selected
            while ri in ri_list:
                ri = random.randrange(1, self.adjacency_mat.shape[0]-1)
            ri_list.append(ri)

            # Swap the ri-th and ri+1-th cities
            tour = population[i, :].copy()
            tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

            # Evaluate the new tour
            new_obj = self.objf(tour)

            # Check if the new tour is better
            if new_obj < best_obj:
                best_tour = tour
                best_obj = new_obj

        # Check if the best tour is different from the original one
        if not np.array_equal(best_tour, population[i, :]):
            modified += 1
            population[i, :] = best_tour
    return population
    

def one_opt_mining(self, population: np.ndarray, k):
    if k < 1 or k > self.adjacency_mat.shape[0] - 1:
        raise ValueError("k must be between 2 and n-1")

    for i in range(population.shape[0]):
        # For each individual in the population
        best_tour = population[i, :]
        best_obj = self.objf(best_tour)

        for j in range(self.adjacency_mat.shape[0] - 1):
            tour = population[i, :].copy()
            tour = tour.astype(int)
        
            # Swap the j-th and j+1-th cities
            tour[j], tour[j+1] = tour[j+1], tour[j]

            if not self.tourIsValid(tour):
                raise ValueError("Duplicate in tour durin two opt")
            # Evaluate the new tour
            new_obj = self.objf(tour)
            # Check if the new tour is better
            if new_obj < best_obj:
                best_tour = tour
                best_obj = new_obj 
        # Check if the best tour is different from the original one
        if not np.array_equal(best_tour, population[i, :]):
            population[i, :] = best_tour
    return population
