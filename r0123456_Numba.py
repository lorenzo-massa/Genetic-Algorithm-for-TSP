import Reporter
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
import itertools
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import ParameterSampler
 

def optimize(verbose=False):


    # Read distance matrix from file.
    #file = open(filename)
    #distanceMatrix = np.loadtxt(file, delimiter=",")
    #file.close()

    # Your code here.    

    # Initialize the population
    population = initialize()

    # Try to improve the initial population with local search
    #population = TSP.two_opt(population, 5)
    iteration = 0

    # Store the progress
    meanObjective = 0.0
    bestObjective = 0.0
    bestSolution = np.zeros(n_cities + 1)

    meanObjectiveList = []
    bestObjectiveList = []
    bestSolutionList = []
    alphaMeanList = []
    variancePopulationList = []

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

        # Join the population and the offspring
        joinedPopulation = np.vstack((population, offspring))

        # Mutation on the joined population without the 10 best solutions
        sorted_population = joinedPopulation[np.argsort(objf_pop(joinedPopulation))]
        joinedPopulation = np.vstack((sorted_population[:10, :], mutation(sorted_population[10:, :])))

        # Local search
        if differentBestSolutions > 0 and differentBestSolutions % 8 == 0:
            tuning += 1

        if tuning > 7:
            tuning = 3
            
        if iteration % 2 == 0:                          # Better not to do to all the population ??
            joinedPopulation = one_opt(joinedPopulation, tuning*2)
        else:
            joinedPopulation = local_search_best_subset(joinedPopulation, tuning)


        # Elimination
        population = elimination_pro(joinedPopulation)


        # Compute progress
        fvals = objf_pop(population)                    # Compute the objective function value for each individual
        meanObjective = np.mean(fvals)                  # Compute the mean objective value of the population
        previousBestObjective = bestObjective           # Store the previous best objective
        bestObjective = np.min(fvals)                   # Store the new best objective
        bestSolution = population[np.argmin(fvals), :]  # Store the new best solution

        # Mean of alpha values
        alphaMean = np.mean(population[:, -1])

        # Adapt alpha value based on the variance of the population
        # variancePopulation = variance(population)
        # adapt_alpha(population)

        # Save progress
        meanObjectiveList.append(meanObjective)
        bestObjectiveList.append(bestObjective)
        bestSolutionList.append(bestSolution)
        alphaMeanList.append(alphaMean)
        variancePopulationList.append(variancePopulation)

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
        timeLeft = reporter.report(meanObjective, bestObjective, bestSolution)

        # Print progress
        if iteration % 5 == 0 and verbose == True:
            print(
                "Iteration: %d, Mean: %f, Best: %f, Alpha: %f, Tuning: %d, Diff. Best: %d, Variance: %f, Time left: %f"
                % (iteration, meanObjective, bestObjective, alphaMean, tuning, differentBestSolutions, variancePopulation, timeLeft)
            )

        if timeLeft < 0:
            yourConvergenceTestsHere = False
            print("Terminated because of time limit")
            break

    # Your code here.
        
    if verbose:

        # Plot results: Mean and best in a figure, alpha and variance in another figure
        t = np.arange(0, iteration, 1)
        fig, axs = plt.subplots(2, 2)

        # Set the dimensions of the figure
        fig.set_figheight(10)
        fig.set_figwidth(15)

        axs[0, 0].plot(t, meanObjectiveList, color="skyblue")
        axs[0, 0].plot(t, bestObjectiveList,  color="lightcoral")
        axs[0, 0].set_title('Mean and best objective function value')
        axs[0, 0].set(xlabel='Iteration', ylabel='Objective function value')
        axs[0, 0].legend(['Mean', 'Best'], loc='upper right')

        axs[0, 1].plot(t, alphaMeanList)
        axs[0, 1].set_title('Mean of alpha values')
        axs[0, 1].set(xlabel='Iteration', ylabel='Alpha value')

        axs[1, 0].plot(t, variancePopulationList)
        axs[1, 0].set_title('Variance of the population')
        axs[1, 0].set(xlabel='Iteration', ylabel='Variance')

        #axs[1, 1].plot(t, alphaMeanList)
        #axs[1, 1].plot(t, variancePopulationList)
        axs[1, 1].set_title('x')
        axs[1, 1].set(xlabel='Iteration', ylabel='x')
        axs[1, 1].legend(['x', 'x'], loc='upper right')

        plt.show()

        # Print the best solution
        print("Best solution: ", bestSolutionList[-1][:-1], " with objective function value: ", bestObjectiveList[-1])

    return bestSolutionList[-1][:-1]   , bestObjectiveList[-1]
 
def optimize_island(verbose=False, testMode=False):

    # Set up the executor
    executor = ProcessPoolExecutor(max_workers=n_population)

    start_time = time.time()

    # Initialize the n_population populations  
    populations = []
    for i in range(n_population):
        populations.append(initialize(alphaList[i]))
        populations[i] = local_search_best_subset(populations[i], 3)

    #print("Initialization time: ", time.time() - start_time)

    # Store the progress of each population
    meanObjectiveList = [[] for i in range(n_population)]
    bestObjectiveList = [[] for i in range(n_population)]
    bestSolutionList = [[] for i in range(n_population)]
    alphaMeanList = [[] for i in range(n_population)]

    # Termination criteria
    differentBestSolutions = [0] * n_population

    # Tuning parameter for the local search
    tuning = [2] * n_population

    # Set up the loop
    iteration = [0] * n_population
    yourConvergenceTestsHere = True

    
    while yourConvergenceTestsHere:

        populations = list(executor.map(process_island, populations, iteration, tuning, k_for_selection, range(n_population)))

        # For each population
        for i in range(n_population):

            #populations[i] = process_island(populations[i], i, iteration[i], tuning[i], differentBestSolutions[i])
            

            # Compute and save progress
            fvals = objf_pop(populations[i])                                    # Compute the objective function value for each individual
            meanObjectiveList[i].append(np.mean(fvals))                         # Compute the mean objective value of the population           
            bestObjectiveList[i].append(np.min(fvals))                          # Store the new best objective
            bestSolutionList[i].append(populations[i][np.argmin(fvals), :])     # Store the new best solution
            alphaMeanList[i].append(np.mean(populations[i][:, -1]))             # Mean of alpha values

            # Adaptation of parameters
            if iteration[i]> 1 and bestObjectiveList[i][-1] == bestObjectiveList[i][-2] and bestObjectiveList[i][-1] != np.inf:
                differentBestSolutions[i] += 1
            else:
                differentBestSolutions[i] = 0

            if differentBestSolutions[i] > 0 and differentBestSolutions[i] % 8 == 0:
                tuning[i] += 1

            if tuning[i] > 6:
                tuning[i] = 3

            iteration[i] += 1

        # Interact the populations 
        if max(iteration) % iterationIneraction == 0:
            # Join the n_population populations
            completePopulation = np.vstack(populations)
            # Shuffle the complete population
            np.random.shuffle(completePopulation)
            # Create n_population from the complete population
            for i in range(n_population):
                populations[i] = completePopulation[i*lambda_:(i+1)*lambda_, :]


        # Find the best solution among the n_population populations
        bestSolution = bestSolutionList[0][-1][:-1]
        bestObjective = bestObjectiveList[0][-1]
        meanObjective = meanObjectiveList[0][-1]
                
        for i in range(1,n_population):
            if bestObjectiveList[i][-1] < bestObjective:
                bestObjective = bestObjectiveList[i][-1]
                bestSolution = bestSolutionList[i][-1][:-1]
                meanObjective = meanObjectiveList[i][-1]

        # Call the reporter with:
        #  - the mean objective function value of the population
        #  - the best objective function value of the population
        #  - a 1D numpy array in the cycle notation containing the best solution
        #    with city numbering starting from 0
        timeLeft = reporter.report(meanObjective, bestObjective, bestSolution)

        # Print progress
        if max(iteration) % 5 == 0 and verbose == True:
            print(
                "Iteration: %d, Mean: %f, Best: %f, Time left: %f, Diff Best: %d"
                % (max(iteration), meanObjective, bestObjective, timeLeft, min(differentBestSolutions))
            )

        # Check for termination
        if max(iteration) >= max_iterations:
            yourConvergenceTestsHere = False
            if not testMode:
                print("Terminated because of max iterations")

        if min(differentBestSolutions) >= MAX_DIFFERENT_BEST_SOLUTIONS:
            yourConvergenceTestsHere = False
            if not testMode:
                print(
                    "Terminated because of %d equal best solutions"
                    % max(differentBestSolutions)
                )

        if timeLeft < 0 and not testMode:
            yourConvergenceTestsHere = False
            if not testMode:
                print("Terminated because of time limit")
            break
        elif (time.time() - start_time) > 300 and testMode:
            yourConvergenceTestsHere = False
            if not testMode:
                print("Terminated because of time limit")
            break
            


    if verbose:
        
        # Plot the mean and the best obj comparing the n_population populations
        t = [np.arange(0, iteration[i], 1) for i in range(n_population)]

        fig, axs = plt.subplots(1, 2)

        # Set the dimensions of the figure
        fig.set_figheight(10)
        fig.set_figwidth(15)

        for i in range(n_population):
            #axs[0].plot(t[i], meanObjectiveList[i])
            axs[0].plot(t[i], bestObjectiveList[i])
            axs[0].set_title('Best objective function value')
            axs[0].set(xlabel='Iteration', ylabel='Objective function value')

            axs[1].plot(t[i], alphaMeanList[i])
            axs[1].set_title('Mean of alpha values')
            axs[1].set(xlabel='Iteration', ylabel='Alpha value')

            #Draw a vertical line every iterationIneraction iterations
            for j in range(1,iteration[i]//iterationIneraction + 1):
                axs[0].axvline(x=iterationIneraction*j, color='grey', linestyle='--')
                axs[1].axvline(x=iterationIneraction*j, color='grey', linestyle='--')

        plt.show()

    # Return the best solution
    return bestSolution, bestObjective

def process_island(population, iteration, tuning, k, i_population):

    # Selection
    selected = selection_roulette(population)

    # Crossover
    offspring = crossover(selected)

    # Join the population and the offspring
    joinedPopulation = np.vstack((population, offspring))

    # Mutation on the joined population without the 10 best solutions
    sorted_population = joinedPopulation[np.argsort(objf_pop(joinedPopulation))]
    joinedPopulation = np.vstack((sorted_population[:10, :], mutation(sorted_population[10:, :], alphaList[i_population])))

    # Local search
    if iteration % 2 == 0:
        joinedPopulation = one_opt(joinedPopulation, tuning)
    else:
        joinedPopulation = local_search_best_subset(joinedPopulation, tuning)

    # Elimination
    population = elimination_pro(joinedPopulation)

    return population

@njit
def tourIsValid(tour: np.ndarray):
    # Check if the tour is valid
    tour = tour.astype(np.int16)
    if len(np.unique(tour[:n_cities])) != n_cities:
        return False
    else:
        return True

def initialize(my_alpha: float) -> np.ndarray:
    # Create a matrix of random individuals
    population = np.zeros((lambda_, n_cities + 1)) # +1 for the alpha value

    not_inf = 0

    for i in range(lambda_):

        if i < lambda_*0.005:
            new_individual = generate_individual_greedy()
        elif i >= lambda_*0.005 and i < lambda_*0.01:
            new_individual = generate_individual_greedy_reverse()
        elif i >= lambda_*0.01 and i < lambda_*0.02:
            new_individual = generate_individual_nearest_neighbor()
        elif i >= lambda_*0.02 and i < lambda_*0.04:
            new_individual = generate_individual_nearest_neighbor_more_index(3)
        elif i >= lambda_*0.04 and i < lambda_*0.06:
            new_individual = generate_individual_nearest_neighbor_more_index(5)
        elif i >= lambda_*0.06 and i < lambda_*0.08:
            new_individual = generate_individual_nearest_neighbor_more_index(7)
        elif i >= lambda_*0.08 and i < lambda_*0.10:
            new_individual = generate_individual_nearest_neighbor_more_index(10)
        else:
            new_individual = generate_individual_random()

        # Evaluate the individual with the objective function
        obj = objf(new_individual)
        # Check if the individual is valid for at most n times
        max_tries = n_cities
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
        alpha = np.array([np.random.normal(my_alpha, 0.03)])
        # Concatenate the alpha value to the individual
        new_individual = np.concatenate((new_individual, alpha))

        if obj == objf(new_individual[:-1]) != np.inf:
            not_inf += 1

        population[i, :] = new_individual


    #print("not_inf: ", not_inf)

    return population

@njit
def objf_pop(population : np.ndarray):
    sum_distance = np.zeros(population.shape[0])

    for i in range(population.shape[0]):
        sum_distance[i] = objf(population[i,:n_cities])
    
    return sum_distance

@njit
def objf(tour : np.ndarray):

    #if not tourIsValid(tour):
    #    print("tour: ", tour)
    #    raise ValueError("Invalid tour during objf")

    #if tour.shape[0] != n_cities:
    #    raise ValueError("The number of cities must be equal to the number of rows of the distance matrix")

    # Convert float64 indices to integers
    tour = tour.astype(np.int16)

    # Apply the objective function to each row of the cities array
    sum_distance = 0

    for ii in range(tour.shape[0] - 1):
        # Sum the distances between the cities
        sum_distance += distanceMatrix[tour[ii], tour[ii + 1]]

        if sum_distance == np.inf:
            return np.inf

    # Add the distance between the last and first city
    sum_distance += distanceMatrix[tour[- 1], tour[0]]

    return sum_distance

@njit
def selection(population: np.ndarray, k: int = 3):

    # Select  parents with elitism
    elite_dim = mu//8
    elite = np.zeros((elite_dim, n_cities + 1))  # +1 for the alpha value
    elite[:,:-1] = elite[:,:-1].astype(np.int16)

    # Select the best individuals
    elite[:elite_dim, :] = population[np.argsort(objf_pop(population))[:elite_dim], :]

    # Create a matrix of selected parents
    selected = np.zeros((mu - elite_dim, n_cities + 1))  # +1 for the alpha value
    selected[:,:-1] = selected[:,:-1].astype(np.int16)

    # Selecting the parents
    for ii in range(mu - elite_dim):
        # Select k random individuals from the population
        ri = np.random.choice(np.arange(1, population.shape[0] - 1), k, replace=False)

        # Select the best individual from the k random individuals
        best = np.argmin(objf_pop(population[ri, :]))

        # Add the selected individual to the matrix of selected parents
        selected[ii, :] = population[ri[best], :]

    # Join the elite and the selected parents
    selected = np.vstack((elite, selected))

    if selected.shape[0] != mu:
        raise ValueError("The number of selected parents must be equal to mu")

    return selected
    

def selection_roulette(population: np.ndarray):

    # Calculate the fitness values
    fitness_values = 1.0 / objf_pop(population)

    # Normalize the fitness values to create probabilities
    probabilities = fitness_values / np.sum(fitness_values)

    # Perform roulette wheel selection
    selected_indices = np.random.choice(np.arange(population.shape[0]), size=mu, p=probabilities, replace=True)

    # Create the matrix of selected parents
    selected = population[selected_indices, :]

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

    tour = tour.astype(np.int16)
    
    # Generate one random index
    ri = np.random.randint(1, n_cities - k)

    # Find the best subset solution from ri to ri+k using brute force

    # Initialize the best tour and objective function value
    best_tour = tour.copy()
    best_obj = objf_perm(tour[ri-1:ri+k+1])

    # Generate all the possible permutations of the cities from ri to ri+k
    if k <= 6:
        permutations = np.array(list(itertools.permutations(tour[ri:ri+k]))) # Faster with small k
    else:
        permutations = generate_permutations(tour[ri:ri+k]) # Numba version
    
    # Add tour[ri-1] and tour[ri+k] to the permutations
    permutations = np.concatenate((np.ones((permutations.shape[0], 1)).astype(np.int16) * tour[ri-1], permutations), axis=1)
    permutations = np.concatenate((permutations, np.ones((permutations.shape[0], 1)).astype(np.int16) * tour[ri+k]), axis=1)

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
        
        if objf(tour) < objf(best_tour):
            print("objf(tour): ", objf(tour))
            print("objf(best_tour): ", objf(best_tour))
            raise ValueError("Best subset solution is worse than the original tour")

    return best_tour

@njit
def _factorial(n):
    if n == 1:
        return n
    else:
        return n * _factorial(n-1)

@njit
def generate_permutations(arr: np.ndarray):
    d = arr.shape[0]
    d_fact = _factorial(d)
    c = np.zeros(d, dtype=np.int32)
    res = np.zeros(shape=(d_fact, d))
    counter = 0
    i = 0
    res[counter] = arr
    counter += 1
    while i < d:
        if c[i] < i:
            if i % 2 == 0:
                arr[0], arr[i] = arr[i], arr[0]
            else:
                arr[c[i]], arr[i] = arr[i], arr[c[i]]
            # Swap has occurred ending the for-loop. Simulate the increment of the for-loop counter
            c[i] += 1
            res[counter] = arr
            counter += 1
            # Simulate recursive call reaching the base case by bringing setting i to the base case
            i = 0
        else:
            # Calling the func(i+1, A) has ended as the for-loop terminated. 
            # Reset the state and increment i.
            c[i] = 0
            i += 1
    # Return array of d! rows and d columns (all permutations)
    return res   

@njit
def objf_perm(tour: np.ndarray):
        
        # Convert float64 indices to integers
        tour = tour.astype(np.int16)
    
        # Apply the objective function to each row of the cities array
        sum_distance = 0
    
        for ii in range(tour.shape[0] - 1):
            # Sum the distances between the cities
            sum_distance += distanceMatrix[tour[ii], tour[ii + 1]]

            if sum_distance == np.inf:
                return np.inf
    
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
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0

    not_visited = np.arange(1, n_cities)

    for ii in range(1, n_cities):
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
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0

    # Last city is random
    individual[n_cities-1] = np.random.randint(1, n_cities)

    not_visited = np.arange(1, n_cities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[n_cities-1])[0][0])

    for ii in range(n_cities-2, 0 , -1):
        # Select the nearest city backwards
        nearest_city = np.argmin(distanceMatrix[not_visited, individual[ii]])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual

@njit
def generate_individual_nearest_neighbor():
    # Create an individual choosing always the nearest city , second city is random

    # Create an individual starting from zero
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0

    # Second city is random from the city accessible from the first city 
    accessible_cities = np.argsort(distanceMatrix[0, :])
    individual[1] = accessible_cities[np.random.randint(1, accessible_cities.shape[0])]

    not_visited = np.arange(1, n_cities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[1])[0][0])

    for ii in range(2, n_cities):
        # Select the nearest city
        nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)

    return individual

@njit
def generate_individual_nearest_neighbor_more_index(k: int):
    # Create an individual choosing always the nearest city , but choosing randomly each k cities
    k = (n_cities + 1)//k

    # Create an individual starting from zero
    individual = np.zeros(n_cities).astype(np.int16)
    individual[0] = 0
    # Second city is random from the city accessible from the first city
    accessible_cities = np.argsort(distanceMatrix[0, :])
    individual[1] = accessible_cities[np.random.randint(1, accessible_cities.shape[0])]

    not_visited = np.arange(1, n_cities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[1])[0][0])

    for ii in range(2, n_cities):

        if ii % k == 0:
            # Select randomly a city from the not visited list
            nearest_city= np.random.randint(0, not_visited.shape[0])
        else:
            # Select the nearest city
            nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])

        # Add the nearest city to the individual
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)


    return individual

@njit
def generate_individual_random():
    r = np.zeros(n_cities).astype(np.int16)
    r[0] = 0
    r[1:] = np.random.permutation(np.arange(1, n_cities)).astype(np.int16)

    return r

@njit
def pmx(parent1: np.ndarray, parent2: np.ndarray):

    # Create a child
    child1 = np.ones(n_cities).astype(np.int16)
    child2 = np.ones(n_cities).astype(np.int16)

    child1 = child1 * -1
    child2 = child2 * -1

    # select random start and end indices for parent1's subsection
    start, end = sorted(np.random.choice(np.arange(1, n_cities - 1), 2, replace=False))

    # copy parent1's subsection into child
    child1[start:end] = parent1[start:end]

    # fill the remaining positions in order
    child1[child1 == -1] = [i for i in parent2[:-1] if i not in child1]

    # copy parent2's subsection into child
    child2[start:end] = parent2[start:end]
    # fill the remaining positions in order
    child2[child2 == -1] = [i for i in parent1[:-1] if i not in child2]

    # Add the alpha value to the child (average of the parents' alpha values)
    new_alpha1 = np.array([parent1[-1]])
    new_alpha2 = np.array([parent2[-1]])

    # Concatenate the child to the alpha value
    child1 = np.concatenate((child1, new_alpha1))
    child2 = np.concatenate((child2, new_alpha2))

    return child1, child2


def pmx2(parent1: np.ndarray, parent2: np.ndarray):
    # Create two children
    child1 = np.ones(n_cities).astype(np.int16)
    child2 = np.ones(n_cities).astype(np.int16)

    # Choose two random crossover points
    point1, point2 = sorted(np.random.choice(np.arange(n_cities), 2, replace=False))

    # Copy the genes between the crossover points from the parents to the children
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]

    # Create mapping arrays for the genes outside the crossover points
    mapping1 = -1 * np.ones(n_cities, dtype=np.int16)
    mapping2 = -1 * np.ones(n_cities, dtype=np.int16)

    # Fill in the mapping arrays based on the crossover points
    mapping1[point1:point2] = parent2[point1:point2]
    mapping2[point1:point2] = parent1[point1:point2]

    # Perform the PMX crossover for the remaining genes
    for i in range(n_cities):
        if mapping1[i] == -1:
            current_gene = parent1[i]
            while current_gene in mapping2:
                current_gene = mapping2[np.where(parent1 == current_gene)]
            child1[i] = current_gene

        if mapping2[i] == -1:
            current_gene = parent2[i]
            while current_gene in mapping1:
                current_gene = mapping1[np.where(parent2 == current_gene)]
            child2[i] = current_gene

    # Add the alpha value to the child
    new_alpha1 = np.array([parent1[-1]])
    new_alpha2 = np.array([parent2[-1]])

    # Concatenate the child to the alpha value
    child1 = np.concatenate((child1, new_alpha1))
    child2 = np.concatenate((child2, new_alpha2))

    if not tourIsValid(child1[:-1]):
        raise ValueError("Invalid tour during pmx crossover")
    if not tourIsValid(child2[:-1]):
        raise ValueError("Invalid tour during pmx crossover")

    return child1, child2


@njit
def scx(parent1: np.ndarray, parent2: np.ndarray):

    # Create a child
    child1 = np.ones(n_cities).astype(np.int16)
    child2 = np.ones(n_cities).astype(np.int16)

    child1 = child1 * -1
    child2 = child2 * -1

    child1[0] = 0
    child2[0] = 0

    # Helper function to find the next city in the sequence
    def find_next_city(parent: np.ndarray, child: np.ndarray, idx):
        current_city = int(child[idx])
        remaining_cities = np.array([i for i in parent[:-1] if i not in child]).astype(np.int16)
        distances = np.array([distanceMatrix[current_city, city] for city in remaining_cities])
        next_city = remaining_cities[np.argmin(distances)]
        return next_city

    # Construct the rest of the child sequences
    for idx in range(1, n_cities):
        child1[idx] = find_next_city(parent1, child1, idx)
        child2[idx] = find_next_city(parent2, child2, idx)

    # Add the alpha value to the child (average of the parents' alpha values)
    new_alpha1 = np.array([parent1[-1]])
    new_alpha2 = np.array([parent2[-1]])

    # Concatenate the child to the alpha value
    child1 = np.concatenate((child1, new_alpha1))
    child2 = np.concatenate((child2, new_alpha2))

    if not tourIsValid(child1[:-1]):
        raise ValueError("Invalid tour during scx crossover")
    if not tourIsValid(child2[:-1]):
        raise ValueError("Invalid tour during scx crossover")

    return child1, child2

@njit
def uox(parent1: np.ndarray, parent2: np.ndarray):
    # Create two children
    child1 = np.ones(n_cities).astype(np.int16)
    child2 = np.ones(n_cities).astype(np.int16)

    child1 = child1 * -1
    child2 = child2 * -1

    child1[0] = 0
    child2[0] = 0


    # Choose two random crossover points
    point1, point2 = sorted(np.random.choice(np.arange(1, n_cities - 1), 2, replace=False))

    # Copy the genes from the first parent between the crossover points
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]

    # Fill in the remaining positions with genes from the second parent in order
    remaining_positions1 = [i for i in parent2[:-1] if i not in child1]
    remaining_positions2 = [i for i in parent1[:-1] if i not in child2]

    idx1 = 0
    idx2 = 0
    for i in range(n_cities):
        if child1[i] == -1:
            child1[i] = remaining_positions1[idx1]
            idx1 += 1
        if child2[i] == -1:
            child2[i] = remaining_positions2[idx2]
            idx2 += 1

        

    # Add the alpha value to the children (average of the parents' alpha values)
    new_alpha1 = np.array([parent1[-1]])
    new_alpha2 = np.array([parent2[-1]])

    # Concatenate the children to the alpha values
    child1 = np.concatenate((child1, new_alpha1))
    child2 = np.concatenate((child2, new_alpha2))

    if not tourIsValid(child1[:-1]):
        raise ValueError("Invalid tour during uox crossover")
    if not tourIsValid(child2[:-1]):
        raise ValueError("Invalid tour during uox crossover")

    return child1, child2

@njit
def crossover(selected: np.ndarray):
    # Create a matrix of offspring
    offspring = np.zeros((mu, n_cities + 1)) # +1 for the alpha value
    offspring[:,:-1] = offspring[:,:-1].astype(np.int16)

    for ii in range(lambda_):
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambda_), 2, replace=False))

        # Perform crossover
        #offspring[ii, :], offspring[ii + lambda_, :] = pmx(selected[ri[0], :], selected[ri[1], :])
        #offspring[ii, :], offspring[ii + lambda_, :] = pmx2(selected[ri[0], :], selected[ri[1], :])
        #offspring[ii, :], offspring[ii + lambda_, :] = scx(selected[ri[0], :], selected[ri[1], :]) TOO EXPENSIVE
        #offspring[ii, :], offspring[ii + lambda_, :] = uox(selected[ri[0], :], selected[ri[1], :])

        # Chose randomly the crossover operator
        #crossover_operator = np.random.choice([pmx, pmx2, scx, uox], p=[0.4, 0.1, 0.1, 0.4])
        #offspring[ii, :], offspring[ii + lambda_, :] = crossover_operator(selected[ri[0], :], selected[ri[1], :])
        if np.random.rand() < 0.51:
            offspring[ii, :], offspring[ii + lambda_, :] = pmx(selected[ri[0], :], selected[ri[1], :])
        else:
            offspring[ii, :], offspring[ii + lambda_, :] = uox(selected[ri[0], :], selected[ri[1], :])

    return offspring

@njit
def swap_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    tour[ri[0]], tour[ri[1]] = tour[ri[1]], tour[ri[0]]

    return tour
    
@njit
def insert_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    removed = tour[ri[1]]
    tour = np.delete(tour, ri[1])
    tour = np.concatenate((tour[:ri[0]], np.array([removed]), tour[ri[0]:]))

    return tour

@njit
def scramble_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    np.random.shuffle(tour[ri[0]:ri[1]])

    return tour

@njit
def inversion_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 2, replace=False))

    tour[ri[0]:ri[1]] = tour[ri[0]:ri[1]][::-1]

    return tour

@njit
def thrors_mutation(tour):
    ri = sorted(np.random.choice(np.arange(1, n_cities), 3, replace=False))

    tour[ri[0]], tour[ri[1]], tour[ri[2]] = tour[ri[1]], tour[ri[2]], tour[ri[0]]

    if not tourIsValid(tour):
        print("tour: ", tour)
        raise ValueError("Invalid tour during thrors mutation") 

    return tour

def mutation(offspring: np.ndarray, alpha: float = 0.6):

    # Apply the mutation to each row of the offspring array
    for ii, _ in enumerate(offspring):
        # Update alpha based on the success of the individual
        #adapt_alpha(offspring[ii, :])

        # Apply the mutation with probability alpha of the individual
        if np.random.rand() < offspring[ii, n_cities]:
            
        # Apply the mutation with probability alpha
        #if np.random.rand() < alpha:

            # Add a noise to the alpha value
            offspring[ii, -1] = np.random.normal(offspring[ii, -1], 0.02)
                        
            # Randomly select a mutation operator with different probabilities
            mutation_operator = np.random.choice([inversion_mutation,swap_mutation, scramble_mutation,
                insert_mutation], p=[0.55, 0.35, 0.05, 0.05]) 
            #mutation_operator = np.random.choice([inversion_mutation,swap_mutation, thrors_mutation], p=[0.50, 0.30, 0.20]) 
            
            offspring[ii, :] = mutation_operator(offspring[ii, :])

            #if not tourIsValid(offspring[ii, :-1]):
            #    print("offspring: ", offspring[ii, :-1])
            #    raise ValueError("Invalid tour during mutation")

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
    

def elimination_pro(joinedPopulation: np.ndarray):

    # Apply the objective function to each row of the joinedPopulation array
    fvals = objf_pop(joinedPopulation)

    # Sort the individuals based on their objective function value
    perm = np.argsort(fvals)

    # Select the best lambda/4 individuals
    n_best = int(lambda_/8)
    best_survivors = joinedPopulation[perm[0 : 3*n_best], :]

    # Select randomly 2*lambda/4 individuals
    random_survivors = joinedPopulation[np.random.choice(perm[n_best:], 4*n_best, replace=False), :]

    #Remove duplicates
    random_survivors = np.unique(random_survivors, axis=0)

    # Generate lambda/4 individuals randomly
    generate_survivors = np.zeros((lambda_ - best_survivors.shape[0] - random_survivors.shape[0], n_cities + 1))
    for i in range(generate_survivors.shape[0]):
        generate_survivors[i, :-1] = generate_individual_random()
        # Take a value from alphaList
        generate_survivors[i, -1] = np.random.normal(alphaList[0], 0.03)

    # Concatenate the best and random survivors
    survivors = np.vstack((best_survivors, random_survivors, generate_survivors))

    if survivors.shape[0] != lambda_:
        raise ValueError("The number of survivors must be equal to lambda_")
    
    return survivors


########################################################################################
        

# Modify the class name to match your student number.
reporter = Reporter.Reporter("r0123456")


lambda_=100 # x 4
mu=lambda_*2
alphaList = np.array([0.7, 0.7, 0.7, 0.7]) 
k_for_selection = np.array([2, 3, 4, 5])
max_iterations=2000
MAX_DIFFERENT_BEST_SOLUTIONS = max_iterations // 10
variancePopulation = 0
n_population = 4


file_path = "tour50.csv"
file = open(file_path)
distanceMatrix = np.loadtxt(file, delimiter=",")
n_cities = distanceMatrix.shape[0]
file.close()

iterationIneraction = 50

results = []

# Main 
if __name__ == "__main__":
    print("File: ", file_path)


    start_time = time.time()
    best_tour, best_obj =  optimize_island(verbose=True)
    end_time = time.time()

    print("Best tour: ", best_tour)
    print("Best objective function value: ", best_obj, " Checked: ", objf(best_tour), " Valid: ", tourIsValid(best_tour))
    print("Time: ", end_time - start_time)


    # Specify the number of iterations
    #n_iter = 10

    #for i in range(n_iter):
        # Run your function
    #    best_tour, best_obj =  optimize_island(verbose=False, testMode=True)

        # Print the results
    #    print("---------------------------------------------")
    #    print("Best objective function value: ", best_obj, " Checked: ", objf(best_tour), " Valid: ", tourIsValid(best_tour))

    #    results.append(best_obj)

    # Define the parameter distributions
    #param_distributions = {
    #    'alphaList': [[np.random.uniform(0.5, 0.85) for _ in range(4)] for _ in range(n_iter)]
    #}

    # Create the parameter sampler
    #sampler = ParameterSampler(param_distributions, n_iter=n_iter, random_state=0)

    # For each combination of parameters
    #for params in sampler:
        # Set the parameters
    #    alphaList = params['alphaList']

        # Print the results
    #    print("---------------------------------------------")
    #    print("Parameters: ", params)

        # Run your function
    #    best_tour, best_obj =  optimize_island(verbose=False, testMode=True)

    #    print("Best objective function value: ", best_obj, " Checked: ", objf(best_tour), " Valid: ", tourIsValid(best_tour))





# Benchmark results to beat:
# PMX
# tour50: simple greedy heuristic 27723     (TARGET 24k)    (BEST 26.2k)    alphaList = np.array([0.7, 0.7, 0.7, 0.7]) k_for_selection = np.array([2, 3, 4, 5])
# tour100: simple greedy heuristic 90851    (TARGET 81k)    (BEST 78.9k)    alphaList = np.array([0.7, 0.7, 0.7, 0.7]) k_for_selection = np.array([2, 3, 4, 5])
# tour200: simple greedy heuristic 39745    (TARGET 35k)    (BEST 37.1k)    alphaList = np.array([0.7, 0.7, 0.7, 0.7]) k_for_selection = np.array([2, 3, 4, 5])
# tour500: simple greedy heuristic 157034   (TARGET 141k)   (BEST 148.5K)   alphaList = np.array([0.7, 0.7, 0.7, 0.7]) k_for_selection = np.array([2, 3, 4, 5])
#                                                           (BEST 148k)     alphaList = np.array([0.55, 0.60, 0.65, 0.70]) k_for_selection = np.array([2, 3, 4, 5])
#                                                                           alphaList = np.array([0.55, 0.55, 0.62, 0.63])  k_for_selection = np.array([2, 3, 4, 5])
# tour750: simple greedy heuristic 197541  (TARGET 177k)   (BEST 192.0k)    alphaList =np.array([0.7, 0.7, 0.7, 0.7]) k_for_selection = np.array([2, 3, 4, 5])
#                                                          (BEST 189.9k)    alphaList =np.array([0.75, 0.75, 0.75, 0.75]) k_for_selection = np.array([2, 3, 4, 5])
# tour1000: simple greedy heuristic 195848 (TARGET 176k)   (BEST 193k)      alphaList = np.array([0.50, 0.55, 0.63, 0.53]) k_for_selection = np.array([2, 3, 4, 4])
#                                                          (BEST 193.7k)    alphaList =np.array([0.7, 0.7, 0.7, 0.7]) k_for_selection = np.array([2, 3, 4, 5])
#                                                          (BEST 187.5k)    alphaList =np.array([0.8, 0.8, 0.8, 0.8]) k_for_selection = np.array([2, 3, 4, 5]) (PMX and UOX)
#                                                          (BEST 192.1k)    alphaList =np.array([0.8, 0.8, 0.8, 0.8]) k_for_selection = np.array([2, 3, 4, 5]) (PMX and UOX) and Roulette Wheel Selection

# To improve:

# Inizializzazione prendendo città con piu infiniti
# Inizializzazione favorendo la copertura di tutto lo spazio di ricerca
# Usa distanceMatrix passato come parametro
# ELIMINAION: crowding
# EGDE RECOMBINATION
