import Reporter
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numba import njit
import itertools
import math

file = "tour500.csv"
distanceMatrix = np.loadtxt(file, delimiter=",")

lambda_=300
mu=lambda_*2
my_alpha=0.25
max_iterations=1000
MAX_DIFFERENT_BEST_SOLUTIONS = max_iterations / 3


# Modify the class name to match your student number.
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        file.close()

        # Your code here.    

        # Initialize the population
        start_time = time.time()
        population = initialize()
        print("--- Initialization: %s seconds ---" % (time.time() - start_time))

        # Try to improve the initial population with local search
        #population = TSP.two_opt(population, 5)
        iteration = 0

        # Store the progress
        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.zeros(distanceMatrix.shape[0] + 1)

        meanObjectiveList = []
        bestObjectiveList = []
        bestSolutionList = []

        # Termination criteria
        previousBestObjective = 0
        differentBestSolutions = 0

        # Local search after initialization
        #population = local_search_best_subset(population, 3)

        tuning = 2

        yourConvergenceTestsHere = True
        while yourConvergenceTestsHere:
            # meanObjective = 0.0
            # bestObjective = 0.0
            # bestSolution = np.array([1,2,3,4,5])

            # Your code here.

            # Perform the evolutionary operators

            # Selection
            selected = selection(population,3)

            # Crossover
            offspring = crossover(selected)

            joinedPopulation = np.vstack((population, mutation(offspring)))

            # Elimination
            if differentBestSolutions == 10:
                tuning += 1
                
            if iteration % 2 == 0: # Better not to do to all the population
                joinedPopulation = one_opt(joinedPopulation, tuning*2)
            else:
                joinedPopulation = local_search_best_subset(joinedPopulation, tuning)

            population = elimination_half_half(joinedPopulation)


            # Show progress
            fvals = objf_pop(population)
            meanObjective = np.mean(fvals)

            previousBestObjective = bestObjective  # Store the previous best objective

            bestObjective = np.min(fvals)
            bestSolution = population[np.argmin(fvals), :]

            # Save progress
            meanObjectiveList.append(meanObjective)
            bestObjectiveList.append(bestObjective)
            bestSolutionList.append(bestSolution)

            # Mean of alpha values
            alphaMean = np.mean(population[:, -1])

            # Print progress
            print(
                "Iteration: %d, Mean: %f, Best: %f, Alpha Mean: %f, Tuning: %d"
                % (iteration, meanObjective, bestObjective, alphaMean, tuning)
            )

            iteration += 1

            # Check for termination
            if iteration >= max_iterations:
                yourConvergenceTestsHere = False

            if bestObjective == previousBestObjective and bestObjective != np.inf:
                differentBestSolutions += 1
            else:
                differentBestSolutions = 0

            if differentBestSolutions >= MAX_DIFFERENT_BEST_SOLUTIONS:
                yourConvergenceTestsHere = False
                print(
                    "Terminated because of %d equal best solutions"
                    % differentBestSolutions
                )

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if(iteration % 50 == 0):
                print("Time left: ", timeLeft)
            if timeLeft < 0:
                yourConvergenceTestsHere = False
                print("Terminated because of time limit")
                break

        # Your code here.

        # Plot results
        t = np.arange(0, iteration, 1)
        plt.plot(t, bestObjectiveList, color="lightcoral")
        plt.plot(t, meanObjectiveList, color="skyblue")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.legend(["Best objective value", "Mean objective value"], loc="upper right")
        plt.show()

        return 0


########################################################################################
    
@njit
def tourIsValid(tour: np.ndarray):
    # Check if the tour is valid
    tour = tour.astype(np.int64)
    if len(np.unique(tour[:distanceMatrix.shape[0]])) != distanceMatrix.shape[0]:
        return False
    else:
        return True
    
@njit
def initialize() -> np.ndarray:
    # Create a matrix of random individuals
    population = np.zeros((lambda_, distanceMatrix.shape[0] + 1)) # +1 for the alpha value

    not_inf = 0

    for i in range(lambda_):

        if i < lambda_*0.01:
            new_individual = generate_individual_greedy()
        elif i >= lambda_*0.01 and i < lambda_*0.02:
            new_individual = generate_individual_greedy_reverse()
        elif i >= lambda_*0.02 and i < lambda_*0.05:
            new_individual = generate_individual_nearest_neighbor()
        else:
            new_individual = generate_individual_random()

        # Evaluate the individual with the objective function
        obj = objf(new_individual)
        # Check if the individual is valid for at most n times
        max_tries = distanceMatrix.shape[0]
        while obj == np.inf and max_tries > 0:
            # Create a random individual
            new_individual = generate_individual_random()
            # Evaluate the individual with the objective function
            obj = objf(new_individual)
            max_tries -= 1

        if not tourIsValid(new_individual):
            print("new_individual: ", new_individual)
            raise ValueError("Invalid tour during initialization")

        # Create alpha with gaussian distribution
        alpha = np.array([np.random.normal(my_alpha, 0.05)])
        # Concatenate the alpha value to the individual
        new_individual = np.concatenate((new_individual, alpha))

        if obj == objf(new_individual[:-1]) != np.inf:
            not_inf += 1

        population[i, :] = new_individual


    print("not_inf: ", not_inf)

    return population

@njit
def objf_pop(population : np.ndarray):
    sum_distance = np.zeros(population.shape[0])

    for i in range(population.shape[0]):
        sum_distance[i] = objf(population[i,:-1])
    
    return sum_distance

@njit
def objf(tour : np.ndarray):

    if not tourIsValid(tour):
        print("tour: ", tour)
        raise ValueError("Invalid tour during objf")

    if tour.shape[0] != distanceMatrix.shape[0]:
        raise ValueError("The number of cities must be equal to the number of rows of the distance matrix")

    # Convert float64 indices to integers
    tour = tour.astype(np.int64)

    # Apply the objective function to each row of the cities array
    sum_distance = 0

    for ii in range(tour.shape[0] - 1):
        # Sum the distances between the cities
        sum_distance += distanceMatrix[tour[ii], tour[ii + 1]]

    # Add the distance between the last and first city
    sum_distance += distanceMatrix[tour[- 1], tour[0]]

    return sum_distance

@njit
def selection(population: np.ndarray, k: int = 3):

    # Create a matrix of selected parents
    selected = np.zeros((mu, distanceMatrix.shape[0] + 1))  # +1 for the alpha value
    selected[:,:-1] = selected[:,:-1].astype(np.int64)

    # Selecting the parents
    for ii in range(mu):
        # Select k random individuals from the population
        ri = np.random.choice(np.arange(1, population.shape[0] - 1), k, replace=False)

        # Select the best individual from the k random individuals
        best = np.argmin(objf_pop(population[ri, :]))

        # Add the selected individual to the matrix of selected parents
        selected[ii, :] = population[ri[best], :]

        if not tourIsValid(selected[ii, :-1]):
            print("selected: ", selected[ii, :-1])
            raise ValueError("Invalid tour during selection")

    return selected

@njit
def one_opt(population: np.ndarray, k):

    if k < 1 or k > population.shape[1] - 1:
        raise ValueError("k must be between 2 and n-1")

    for i in range(population.shape[0]):

        # For each individual in the population
        best_tour = population[i, :]
        best_obj = objf(best_tour[:-1])

        # Select k random indices
        k_list = sorted(np.random.choice(np.arange(1, population.shape[1] - 2), k, replace=False))

        for ri in k_list:
            # Swap the ri-th and ri+1-th cities
            tour = population[i, :].copy()
            tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

            # Evaluate the new tour
            new_obj = objf(tour[:-1])

            # Check if the new tour is better
            if new_obj < best_obj:
                best_tour = tour
                best_obj = new_obj

        population[i, :] = best_tour

    return population

def local_search_best_subset(population: np.ndarray, k: int):
    # Apply the best subset solution to each row of the population array
    for ii in range(population.shape[0]):
        population[ii, :-1] = best_subset_solution(population[ii, :-1], k)
    
    return population

def best_subset_solution(tour: np.ndarray, k: int):

    tour = tour.astype(np.int64)

    # Generate one randdom index
    ri = np.random.choice(np.arange(1, distanceMatrix.shape[0] - 1 - k), 1, replace=False)[0]

    # Find the best subset solution from ri to ri+k using brute force

    # Initialize the best tour and objective function value
    best_tour = tour.copy()
    best_obj = objf_perm(tour[ri:ri+k+1])

    # Generate all the possible permutations of the cities from ri to ri+k
    permutations = np.array(list(itertools.permutations(tour[ri:ri+k])))

    # TODO GENERATE PERMUTATIONS WITH NUMBA
    
    # Add tour[ri-1] and tour[ri+k] to the permutations
    permutations = np.concatenate((np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri-1], permutations), axis=1)
    permutations = np.concatenate((permutations, np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri+k]), axis=1)

    # Evaluate the objective function for each permutation
    objs = objf_permutation(permutations)
    best = np.argmin(objs)

    # Check if the best permutation is better than the original tour
    if objs[best] < best_obj:

        # Update the best tour and objective function value
        best_tour[ri:ri+k] = permutations[best, 1:-1]

        if not tourIsValid(best_tour):
            print("best_tour: ", best_tour)
            raise ValueError("Invalid tour during best subset solution")

    return best_tour

@njit
def calculate_factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

@njit
def generate_permutations(arr: np.ndarray):
    n = arr.shape[0]
    factorial = calculate_factorial(n)
    result = np.empty((factorial, n), dtype=arr.dtype)

    for i in range(factorial):
        pool = list(arr)
        for j in range(n, 0, -1):
            index = i // calculate_factorial(j - 1)
            result[i, n - j] = pool.pop(index)
            i -= index * calculate_factorial(j - 1)

    return result

@njit
def objf_perm(tour: np.ndarray):
        
        # Convert float64 indices to integers
        tour = tour.astype(np.int64)
    
        # Apply the objective function to each row of the cities array
        sum_distance = 0
    
        for ii in range(tour.shape[0] - 1):
            # Sum the distances between the cities
            sum_distance += distanceMatrix[tour[ii], tour[ii + 1]]
    
        return sum_distance

@njit
def objf_permutation(permutations: np.ndarray):
        # Apply the objective function to each row of the permutations array
        obj = np.zeros(permutations.shape[0])
    
        for ii in range(permutations.shape[0]):
            obj[ii] = objf_perm(permutations[ii, :])
    
        return obj

@njit
def generate_individual_greedy():
    # Create an individual choosing always the nearest city
    individual = np.zeros(distanceMatrix.shape[0]).astype(np.int64)
    individual[0] = 0

    not_visited = np.arange(1, distanceMatrix.shape[0])

    for ii in range(1, distanceMatrix.shape[0]):
        # Select the nearest city
        nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual
    
@njit
def generate_individual_greedy_reverse():
    # Create an individual choosing always the nearest city
    individual = np.zeros(distanceMatrix.shape[0]).astype(np.int64)
    individual[0] = 0

    # Last city is random
    individual[distanceMatrix.shape[0]-1] = np.random.randint(1, distanceMatrix.shape[0])

    not_visited = np.arange(1, distanceMatrix.shape[0])
    not_visited = np.delete(not_visited, np.where(not_visited == individual[distanceMatrix.shape[0]-1])[0][0])

    for ii in range(distanceMatrix.shape[0]-2, 0 , -1):
        # Select the nearest city backwards
        nearest_city = np.argmin(distanceMatrix[not_visited, individual[ii]])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)


    if not tourIsValid(individual):
        print("individual: ", individual)
        raise ValueError("Invalid tour during greedy reverse")

    return individual



@njit
def generate_individual_nearest_neighbor():
    # Create an individual choosing always the nearest city , second city is random
    individual = np.zeros(distanceMatrix.shape[0]).astype(np.int64)
    individual[0] = 0
    individual[1] = np.random.randint(1, distanceMatrix.shape[0]) # Second city is random

    not_visited = np.arange(1, distanceMatrix.shape[0])
    not_visited = np.delete(not_visited, np.where(not_visited == individual[1])[0][0])

    for ii in range(2, distanceMatrix.shape[0]):
        # Select the nearest city
        nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual

@njit
def generate_individual_random():
    r = np.zeros(distanceMatrix.shape[0]).astype(np.int64)
    r[0] = 0
    r[1:] = np.random.permutation(np.arange(1, distanceMatrix.shape[0])).astype(np.int64)

    return r

@njit
def pmx(parent1, parent2):

    # Create a child
    child1 = np.ones(distanceMatrix.shape[0]).astype(np.int64)
    child2 = np.ones(distanceMatrix.shape[0]).astype(np.int64)

    child1 = child1 * -1
    child2 = child2 * -1

    # select random start and end indices for parent1's subsection
    start, end = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0] - 1), 2, replace=False))

    # copy parent1's subsection into child
    child1[start:end] = parent1[start:end]

    # fill the remaining positions in order
    child1[child1 == -1] = [i for i in parent2[:-1] if i not in child1]

    # copy parent2's subsection into child
    child2[start:end] = parent2[start:end]
    # fill the remaining positions in order
    child2[child2 == -1] = [i for i in parent1[:-1] if i not in child2]


    if not tourIsValid(child1):
        print("start, end: ", start, end)
        print("parent1: ", parent1)
        print("parent2: ", parent2)
        print("child1: ", child1)
        raise ValueError("Invalid tour during crossover")
    
    if not tourIsValid(child2):
        print("start, end: ", start, end)
        print("parent1: ", parent1)
        print("parent2: ", parent2)
        print("child2: ", child2)
        raise ValueError("Invalid tour during crossover")

    # Add the alpha value to the child (average of the parents' alpha values)
    new_alpha = np.array([(parent1[-1] + parent2[-1]) / 2])

    # Concatenate the child to the alpha value
    child1 = np.concatenate((child1, new_alpha))
    child2 = np.concatenate((child2, new_alpha))

    return child1, child2

@njit
def crossover(selected: np.ndarray):
    # Create a matrix of offspring
    offspring = np.zeros((mu, distanceMatrix.shape[0] + 1)) # +1 for the alpha value
    offspring[:,:-1] = offspring[:,:-1].astype(np.int64)

    for ii in range(lambda_):
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambda_), 2, replace=False))

        # Perform crossover
        offspring[ii, :], offspring[ii + lambda_, :] = pmx(selected[ri[0], :], selected[ri[1], :])

    return offspring

@njit
def swap_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))

    tour[ri[0]], tour[ri[1]] = tour[ri[1]], tour[ri[0]]

    return tour
    
@njit
def insert_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))

    removed = tour[ri[1]]
    tour = np.delete(tour, ri[1])
    tour = np.concatenate((tour[:ri[0]], np.array([removed]), tour[ri[0]:]))

    return tour

@njit
def scramble_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))

    np.random.shuffle(tour[ri[0]:ri[1]])

    return tour

@njit
def inversion_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))

    tour[ri[0]:ri[1]] = tour[ri[0]:ri[1]][::-1]

    return tour


def mutation(offspring: np.ndarray):

    # Apply the mutation to each row of the offspring array
    for ii, _ in enumerate(offspring):
        # Update alpha based on the success of the individual
        #self.adapt_alpha(offspring[ii, :])

        # Apply the mutation with probability alpha of the individual
        if np.random.rand() < offspring[ii, distanceMatrix.shape[0]]:

            # Add a noise to the alpha value
            offspring[ii, -1] = np.random.normal(offspring[ii, -1], 0.02)
                        
            # Randomly select a mutation operator with different probabilities
            mutation_operator = np.random.choice([inversion_mutation,swap_mutation,
                scramble_mutation,insert_mutation], p=[0.55, 0.35, 0.05, 0.05])
            
            #random.choices([inversion_mutation,swap_mutation,
            #    scramble_mutation,insert_mutation], weights=[9, 4, 1, 1], k=1)[0]
            
            offspring[ii, :] = mutation_operator(offspring[ii, :])

            if not tourIsValid(offspring[ii, :-1]):
                print("offspring: ", offspring[ii, :-1])
                raise ValueError("Invalid tour during mutation")

    return offspring

@njit
def elimination( joinedPopulation: np.ndarray):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = objf_pop(joinedPopulation)
        # Sort the individuals based on their objective function value
        perm = np.argsort(fvals)
        # Select the best lambda individuals
        survivors = joinedPopulation[perm[0 : lambda_], :]

        return survivors
    

def elimination_half_half(joinedPopulation: np.ndarray):

    # Apply the objective function to each row of the joinedPopulation array
    fvals = objf_pop(joinedPopulation)

    # Sort the individuals based on their objective function value
    perm = np.argsort(fvals)

    # Select the best lambda/2 individuals
    n_best = int(lambda_/4)
    best_survivors = joinedPopulation[perm[0 : n_best], :]

    # Select randomly the rest individuals
    random_survivors = joinedPopulation[np.random.choice(perm[n_best:], lambda_ - n_best, replace=False), :]

    # Concatenate the best and random survivors
    survivors = np.vstack((best_survivors, random_survivors))
    
    return survivors

########################################################################################


# Main 
if __name__ == "__main__":

    print(generate_permutations(np.array([1,2,3])))


    # Calculate the time
    start_time = time.time()
    r0123456().optimize(file)
    print("--- %s seconds ---" % (time.time() - start_time))


# Benchmark results to beat:
# tour50: simple greedy heuristic 27723     (TARGET 24k)    (BEST 26k)
# tour100: simple greedy heuristic 90851    (TARGET 81k)    (BEST 81k)
# tour200: simple greedy heuristic 39745    (TARGET 35k)    (BEST 38k)
# tour500: simple greedy heuristic 157034   (TARGET 141k)   (BEST 150k)
# tour750: simple greedy heuristic 197541  (TARGET 177k)   (BEST 200k)
# tour1000: simple greedy heuristic 195848 (TARGET 176k)   (BEST 200k)
