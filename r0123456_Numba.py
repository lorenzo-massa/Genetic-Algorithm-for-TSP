import Reporter
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numba import njit

file = "tour750.csv"
distanceMatrix = np.loadtxt(file, delimiter=",")

lambda_=300
mu=lambda_*2
my_alpha=0.25
max_iterations=1000


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

        TSP = TravelsalesmanProblem(distanceMatrix,
                                    lambda_=lambda_,
                                    mu=mu,
                                    alpha=my_alpha,
                                    max_iterations=max_iterations)
        

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
            if iteration < 80:
                population = elimination(joinedPopulation)
            elif iteration >= 80 and iteration < 180:
                joinedPopulation = one_opt(joinedPopulation, 3)
                population = elimination(joinedPopulation)
            elif iteration >= 180 and iteration < 300:
                joinedPopulation = one_opt(joinedPopulation, 5)
                population = elimination(joinedPopulation)
            else:
                joinedPopulation = one_opt(joinedPopulation, 20)
                population = elimination(joinedPopulation)


            # Show progress
            fvals = objf_pop(population)
            meanObjective = np.mean(fvals)

            previousBestObjective = bestObjective  # Store the previous best objective

            bestObjective = np.min(fvals)
            TSP.bestObj = bestObjective
            bestSolution = population[np.argmin(fvals), :]

            # Save progress
            meanObjectiveList.append(meanObjective)
            bestObjectiveList.append(bestObjective)
            bestSolutionList.append(bestSolution)

            # Mean of alpha values
            alphaMean = np.mean(population[:, -1])

            # Print progress
            print(
                "Iteration: %d, Mean: %f, Best: %f, Alpha Mean: %f"
                % (iteration, meanObjective, bestObjective, alphaMean)
            )

            iteration += 1

            # Check for termination
            if iteration >= TSP.max_iterations:
                yourConvergenceTestsHere = False

            if bestObjective == previousBestObjective and bestObjective != np.inf:
                differentBestSolutions += 1
            else:
                differentBestSolutions = 0

            if differentBestSolutions >= TSP.MAX_DIFFERENT_BEST_SOLUTIONS:
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


class TravelsalesmanProblem:
    def __init__(
        self,
        adjacency_mat: np.ndarray,
        lambda_: int,
        mu: int,
        alpha: float,
        max_iterations,
    ) -> None:
        self.adjacency_mat = adjacency_mat
        self.lambda_ = lambda_  # Population size
        self.mu = mu  # Offspring size (must be the double of lambda)
        self.alpha = alpha  # Mutation probability
        self.max_iterations = max_iterations  # Maximum number of iterations
        self.MAX_DIFFERENT_BEST_SOLUTIONS = max_iterations / 4
        self.bestObj = np.inf

    def selection(self,population: np.ndarray, k: int = 3):

        # Create a matrix of selected parents
        selected = np.zeros((self.mu, self.adjacency_mat.shape[0] + 1))  # +1 for the alpha value
        selected[:,:-1] = selected[:,:-1].astype(int)

        # Selecting the parents
        for ii in range(self.mu):
            # Select k random individuals from the population
            ri = random.sample(range(population.shape[0]), k)

            print("ri: ", ri)
            print("population[ri]: ", population[ri])

            # Select the best individual from the k random individuals
            best = np.argmin(np.apply_along_axis(objf, 1, population[ri, :-1]))

            # Add the selected individual to the matrix of selected parents
            selected[ii, :] = population[ri[best], :]
        return selected

    
    def elimination_with_crowding(self, joinedPopulation: np.ndarray):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = np.apply_along_axis(objf, 1, joinedPopulation[:, :-1])

        # Calculate the crowding distance for each individual
        crowding_distances = self.crowding_distances(joinedPopulation, fvals)

        # Sort the individuals based on the objective function value and crowding distance
        sorted_indices = np.lexsort((crowding_distances, fvals))
        sorted_population = joinedPopulation[sorted_indices, :]

        # Select the best lambda individuals with crowding selection
        survivors = sorted_population[:self.lambda_, :]

        return survivors

    
    def crowding_distances(self, population: np.ndarray, fvals: np.ndarray):
        num_individuals, _ = population.shape
        crowding_distances = np.zeros(num_individuals)

        # Calculate crowding distance for each individual
        for obj_idx in range(population.shape[1] - 1):  # Exclude the last column (alpha values)
            sorted_indices = np.argsort(fvals)
            crowding_distances[sorted_indices[0]] = np.inf
            crowding_distances[sorted_indices[-1]] = np.inf

            obj_range = fvals[sorted_indices[-1]] - fvals[sorted_indices[0]]
            if obj_range == 0:
                continue

            for i in range(1, num_individuals - 1):
                crowding_distances[sorted_indices[i]] += (
                    fvals[sorted_indices[i + 1]] - fvals[sorted_indices[i - 1]]
                ) / obj_range

        return crowding_distances


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

    for i in range(lambda_):

        if i < lambda_*0.02:
            new_individual = generate_individual_greedy()
        if i >= lambda_*0.02 and i < lambda_*0.05:
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

        population[i, :] = new_individual

    return population

@njit
def objf_pop(population : np.ndarray):
    sum_distance = np.zeros(population.shape[0])

    for i in range(population.shape[0]):
        if not tourIsValid(population[i,:-1]):
            print("population ", i)
            print(population[i])
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
    
def one_opt_mining(population: np.ndarray):

        for i in range(population.shape[0]):

            # For each individual in the population
            best_tour = population[i, :]
            best_obj = objf(best_tour[:-1])


            for j in range(distanceMatrix.shape[0] - 1):
                tour = population[i, :].copy()
                tour[:-1] = tour[:-1].astype(np.int64)
                
                # Swap the j-th and j+1-th cities
                tour[j], tour[j+1] = tour[j+1], tour[j]

                if not tourIsValid(tour[:-1]):
                    print("tour: ", tour)
                    raise ValueError("Duplicate in tour durin two opt")


                # Evaluate the new tour
                new_obj = objf(tour[:-1])

                # Check if the new tour is better
                if new_obj < best_obj:
                    best_tour = tour
                    best_obj = new_obj 


            # Check if the best tour is different from the original one
            if not np.array_equal(best_tour, population[i, :]):
                population[i] = best_tour

        return population


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
    

########################################################################################


# Main 
if __name__ == "__main__":
    # Calculate the time
    start_time = time.time()
    r0123456().optimize(file)
    print("--- %s seconds ---" % (time.time() - start_time))


# Benchmark results to beat:
# tour50: simple greedy heuristic 27723     (TARGET 24k)    (BEST 26k)
# tour100: simple greedy heuristic 90851    (TARGET 81k)    (BEST 81k)
# tour200: simple greedy heuristic 39745    (TARGET 35k)    (BEST 38k)
# tour500: simple greedy heuristic 157034   (TARGET 141k)   (BEST 155k)
# tour750: simple greedy heuristic 197541  (TARGET 177k)   (BEST 203k)
# tour1000: simple greedy heuristic 195848 (TARGET 176k)   (BEST 205k)
