import Reporter
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
import itertools
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
 

# Modify the class name to match your student number.
reporter = Reporter.Reporter("r0123456")

# The evolutionary algorithm's main loop
def optimize(distanceMatrix, verbose=False):


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
        
    if verbose == True:

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
 


def optimize_island(distanceMatrix, verbose=False):

    # Initialize the n_population populations  
    populations = []
    for i in range(n_population):
        populations.append(initialize(alphaList[i]))

        # Local search after initialization
        populations[i] = one_opt(populations[i], 2+i)
    

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

        # For each population
        for i in range(n_population):

            # Selection
            selected = selection(populations[i],k_for_selection[i])

            # Crossover
            offspring = crossover(selected)

            # Join the population and the offspring
            joinedPopulation = np.vstack((populations[i], offspring))

            # Mutation on the joined population without the 10 best solutions
            sorted_population = joinedPopulation[np.argsort(objf_pop(joinedPopulation))]
            joinedPopulation = np.vstack((sorted_population[:5, :], mutation(sorted_population[5:, :])))

            # Local search
            if differentBestSolutions[i] > 0 and differentBestSolutions[i] % 8 == 0:
                tuning[i] += 1

            if tuning[i] > 6:
                tuning[i] = 3
                
            if iteration[i] % 2 == 0:                          # Better not to do to all the population ??
                joinedPopulation = one_opt(joinedPopulation, tuning[i]*(1+i))
            else:
                joinedPopulation = local_search_best_subset(joinedPopulation, tuning[i])

            # Elimination
            populations[i] = elimination_pro(joinedPopulation)


            # Compute and save progress
            fvals = objf_pop(populations[i])                                    # Compute the objective function value for each individual
            meanObjectiveList[i].append(np.mean(fvals))                         # Compute the mean objective value of the population           
                                                                                # Store the previous best objective
            bestObjectiveList[i].append(np.min(fvals))                          # Store the new best objective
            bestSolutionList[i].append(populations[i][np.argmin(fvals), :])     # Store the new best solution
            alphaMeanList[i].append(np.mean(populations[i][:, -1]))             # Mean of alpha values

            if iteration[i]> 1 and bestObjectiveList[i][-1] == bestObjectiveList[i][-2] and bestObjectiveList[i][-1] != np.inf:
                differentBestSolutions[i] += 1
            else:
                differentBestSolutions[i] = 0

            iteration[i] += 1


        # Check for termination
        if max(iteration) >= max_iterations:
            yourConvergenceTestsHere = False

        if max(differentBestSolutions) >= MAX_DIFFERENT_BEST_SOLUTIONS:
            yourConvergenceTestsHere = False
            print(
                "Terminated because of %d equal best solutions"
                % max(differentBestSolutions)
            )

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
                % (max(iteration), meanObjective, bestObjective, timeLeft, max(differentBestSolutions))
            )

        if timeLeft < 0:
            yourConvergenceTestsHere = False
            print("Terminated because of time limit")
            break

        if iteration[0] % iterationIneraction == 0:
            # Join the n_population populations
            completePopulation = np.vstack(populations)
            # Shuffle the complete population
            np.random.shuffle(completePopulation)
            # Create n_population from the complete population
            for i in range(n_population):
                populations[i] = completePopulation[i*lambda_:(i+1)*lambda_, :]

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

        #Draw a vertical line each 50 iterations
        for j in range(1,iteration[i]//iterationIneraction):
            axs[0].axvline(x=50*j, color='grey', linestyle='--')
            axs[1].axvline(x=50*j, color='grey', linestyle='--')

    
    plt.show()



    # Return the best solution
    return bestSolution, bestObjective

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

        if i < lambda_*0.01:
            new_individual = generate_individual_greedy()
        elif i >= lambda_*0.01 and i < lambda_*0.02:
            new_individual = generate_individual_greedy_reverse()
        elif i >= lambda_*0.02 and i < lambda_*0.04:
            new_individual = generate_individual_nearest_neighbor_more_index(5)
        elif i >= lambda_*0.04 and i < lambda_*0.06:
            new_individual = generate_individual_nearest_neighbor()
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
        alpha = np.array([np.random.normal(my_alpha, 0.05)])
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

    # Create a matrix of selected parents
    selected = np.zeros((mu, n_cities + 1))  # +1 for the alpha value
    selected[:,:-1] = selected[:,:-1].astype(np.int16)

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
def pmx(parent1, parent2):

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


    #if not tourIsValid(child1):
    #    print("start, end: ", start, end)
    #    print("parent1: ", parent1)
    #    print("parent2: ", parent2)
    #    print("child1: ", child1)
    #    raise ValueError("Invalid tour during crossover")
    
    #if not tourIsValid(child2):
    #    print("start, end: ", start, end)
    #    print("parent1: ", parent1)
    #    print("parent2: ", parent2)
    #    print("child2: ", child2)
    #    raise ValueError("Invalid tour during crossover")

    # Add the alpha value to the child (average of the parents' alpha values)
    new_alpha = np.array([(parent1[-1] + parent2[-1]) / 2])

    # Concatenate the child to the alpha value
    child1 = np.concatenate((child1, new_alpha))
    child2 = np.concatenate((child2, new_alpha))

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
        offspring[ii, :], offspring[ii + lambda_, :] = pmx(selected[ri[0], :], selected[ri[1], :])

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


def mutation(offspring: np.ndarray):

    # Apply the mutation to each row of the offspring array
    for ii, _ in enumerate(offspring):
        # Update alpha based on the success of the individual
        #adapt_alpha(offspring[ii, :])

        # Apply the mutation with probability alpha of the individual
        if np.random.rand() < offspring[ii, n_cities]:

            # Add a noise to the alpha value
            offspring[ii, -1] = np.random.normal(offspring[ii, -1], 0.02)
                        
            # Randomly select a mutation operator with different probabilities
            mutation_operator = np.random.choice([inversion_mutation,swap_mutation,
                scramble_mutation,insert_mutation], p=[0.55, 0.35, 0.05, 0.05])
            
            #random.choices([inversion_mutation,swap_mutation,
            #    scramble_mutation,insert_mutation], weights=[9, 4, 1, 1], k=1)[0]
            
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
    
@njit
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

    # Generate lambda/4 individuals randomly
    generate_survivors = np.zeros((lambda_ - 7*n_best, n_cities + 1))
    for i in range(lambda_ - 7*n_best):
        generate_survivors[i, :-1] = generate_individual_random()
        # Take a value from alphaList
        generate_survivors[i, -1] = np.random.normal(np.random.choice(alphaList), 0.05)

    # Concatenate the best and random survivors
    survivors = np.vstack((best_survivors, random_survivors, generate_survivors))
    
    return survivors


def variance(population: np.ndarray):
    # Calculate the variance of the population
    num_individuals =  population.shape[0]
    total_distance = 0

    for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                distance = np.linalg.norm(population[i] - population[j])
                total_distance += distance

    
    return total_distance / (num_individuals * (num_individuals - 1) / 2)

@njit
def adapt_alpha(population: np.ndarray):
    # Adapt the alpha value based on the variance of the population

    for ii in range(population.shape[0]):
        # Calculate the new alpha value
        # If the variance of the population is low, increase alpha
        # If the variance of the population is high, decrease alpha
        if variancePopulation < 0.00015:
            new_alpha = population[ii, -1] + 0.01
        elif variancePopulation > 0.00025:
            new_alpha = population[ii, -1] - 0.01
        else:
            new_alpha = population[ii, -1] 


        # Check if the new alpha value is between 0 and 1
        if new_alpha < 0.25:
            new_alpha = 0.25
        elif new_alpha > 0.75:
            new_alpha = 0.75

        # Update the alpha value
        population[ii, -1] = new_alpha




########################################################################################

lambda_=80 # x 4
mu=lambda_*2
alphaList = np.array([0.3, 0.4, 0.5, 0.6])
k_for_selection = np.array([3, 3, 3, 4])
max_iterations=1000
MAX_DIFFERENT_BEST_SOLUTIONS = max_iterations // 5
variancePopulation = 0
iterationIneraction = 50
n_population = 4


file_path = "tour200.csv"
file = open(file_path)
distanceMatrix = np.loadtxt(file, delimiter=",")
n_cities = distanceMatrix.shape[0]
file.close()

results = []

# Main 
if __name__ == "__main__":
    print("File: ", file_path)

    # Create a process pool executor with 2 processes
    #n_processors = 2
    #with Pool(n_processors) as executor:
    #    results.append(executor.map(optimize, [distanceMatrix,distanceMatrix]))

    #print(results)

    #for i in range(n_tries):
    #    print("Try: ", i)
    start_time = time.time()
    best_tour, best_obj =  optimize_island(distanceMatrix, verbose=True)
    end_time = time.time()
    #    # Save the best tour, objective function value
    #    results.append((best_tour, best_obj))
    print("Time: ", end_time - start_time)
    print("Best tour: ", best_tour)
    print("Best objective function value: ", best_obj, " Checked: ", objf(best_tour), " Valid: ", tourIsValid(best_tour))
    # Print all the results
    #for i, result in enumerate(results):
    #    print("Result ", i, ": ", result)



# Benchmark results to beat:
# tour50: simple greedy heuristic 27723     (TARGET 24k)    (BEST 26k)
# tour100: simple greedy heuristic 90851    (TARGET 81k)    (BEST 81k)
# tour200: simple greedy heuristic 39745    (TARGET 35k)    (BEST 38k)
# tour500: simple greedy heuristic 157034   (TARGET 141k)   (BEST 150k)
# tour750: simple greedy heuristic 197541  (TARGET 177k)   (BEST 190k)
# tour1000: simple greedy heuristic 195848 (TARGET 176k)   (BEST 193k)

# To improve:

# Inizializzazione prendendo città con piu infiniti
# Inizializzazione favorendo la copertura di tutto lo spazio di ricerca
# Alpha adaptation fatta per bene
# Elimination con crowding
# 2 selection da provare
# Usa distanceMatrix passato come parametro