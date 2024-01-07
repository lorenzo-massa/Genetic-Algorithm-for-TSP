import numpy as np
import random
import time
import Utilis
import matplotlib.pyplot as plt
import itertools 
import math
import warnings
import Reporter
from numba import jit

class Individual:
    def __init__(self, numCities, path: np.array=None, alpha: float=0.30):
        if path is None:
            self.path = np.zeros(numCities-1).astype(np.int64)
        else:
            self.path = path
        self.alpha = alpha

    def change_alpha(self):
        # self.alpha = max(0.04, 0.20 + 0.08*np.random.randn())
        return np.random.normal(self.alpha, 0.03)


class r0978353:
    """ Parameters """
    def __init__(self):
        self.reporter = Reporter.Reporter(__class__.__name__)

        # Parameters
        self.alpha = 0.3                                     # Mutation probability
        self.mutationratios = [0.9, 0.1, 0, 0]            # inversion, swap, scramble, insert
        self.lambdaa = 150                                    # Population size
        self.mu = self.lambdaa * 2                                 # Offspring size       
        self.k= 3                                            # Tournament selection
        self.numIters = 2000                                  # Maximum number of iterations
        self.objf = fitness                                   # Objective function
        self.maxSameBestSol = 100  
        self.maxTime = 300                                    # Maximum 5 minutes  

    def optimize(self, filename):
        distanceMatrix = read_from_file(filename)
        numCities = distanceMatrix.shape[0]

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Store information at each iteration of the main loop
        mean = []
        best = []
        bestSolutions = []
        alpha_list = []
        i=0
        tuning = 3
        
        # Termination criteria
        previousBestFitness = 0
        countSameBestSol = 0
        bestFitness = 0.0
        bestSolution = np.zeros(numCities - 1)
        terminationCriteria = True  

        # Start of the total time for the selected instance
        startTotTime = time.time()

        # Count time for the initilization step
        startInitialization = time.time()
        population = initialization_mix(distanceMatrix, numCities, self.lambdaa)
        initializationTime = time.time() - startInitialization
        timePassed = initializationTime
        initializationTime = time.time() - startInitialization
        print("--- Initialization: %s seconds ---" % initializationTime)
    
        while terminationCriteria:
            # Selection
            start = time.time()
            selected = selection_kTour(numCities, distanceMatrix, population, self.k, self.mu)  # pop of ind                                                 # selected = initial*2
            # selected = selection_topK(distanceMatrix, numCities, population, len(population)*0.75, self.mu)

            # Recombination
            offspring = pmx_crossover(numCities, distanceMatrix, selected, self.k, self.lambdaa)                                                           # offspring = initial
            # offspring = crossover(distanceMatrix, selected, mu, lambdaa)  
            # offspring = crossover_scx(distanceMatrix, selected, self.lambdaa)   
            # offspring = pmx_crossover_j(distanceMatrix, numCities, selected)            

            # Mutation
            joinedPopulation = np.concatenate((population, offspring))   
            sorted_population = joinedPopulation[np.argsort(pop_fitness(distanceMatrix, joinedPopulation))]
            # to mutate more poulation thaqn just the offspring, but preserve the best 10 solutions
            joinedPopulation = np.concatenate((sorted_population[:10], mutate(distanceMatrix, numCities, sorted_population[10:], self.mutationratios)))

            # Elimination and Local Search
            # population = elimination_withoutLocal(distanceMatrix, joinedPopulation, self.lambdaa) # TODO put local search before elimination
            population = elimination_localSearch(numCities, distanceMatrix, joinedPopulation, countSameBestSol, i, tuning, self.lambdaa, self.mu, self.alpha)

            # Compute fitness of this current iteration
            itT = time.time() - start
            fvals = pop_fitness(distanceMatrix, population)                   
            # Save all progressmutate
            timePassed += (time.time() - start)
            previousBestFitness = bestFitness
            bestFitness = np.min(fvals)
            bestSolution = population[np.argmin(fvals)]
            bestSolutions.append(bestSolution)
            mean.append(np.mean(fvals))
            best.append(np.min(fvals))
            alpha_list += [np.mean(p.alpha) for p in population]

            timeLeft = self.reporter.report(mean[i], best[i], bestSolution.path)
            print(f'{i}) {timePassed: .2f}s:\t Mean fitness = {mean[i]: .2f} \t Best fitness = {best[i]: .2f} \t alpha best ind = {bestSolution.alpha: .2f} \t alphas = {alpha_list[-1]: .2f} \t pop length={population.shape}\t')
            i=i+1

            # Check for termination:
            # Termination criteria: number of iterations
            if i >= self.numIters:
                terminationCriteria = False
                break
            # Termination criteria: bestFitness doesn't improve for 'maxSameBestSol' times
            if bestFitness == previousBestFitness and bestFitness != np.inf:
                countSameBestSol += 1
            else:
                countSameBestSol = 0
            if countSameBestSol >= self.maxSameBestSol:
                terminationCriteria = False
                break

            # Termination criteria: time limit
            if timeLeft < 0:
                terminationCriteria = False
                break       
        return 0  

def read_from_file(filename):
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    return distanceMatrix

# no numba
def initialization_mix(distanceMatrix, numCities, lambdaa) -> np.ndarray:
    population = np.empty(lambdaa, dtype=object)

    for i in range(lambdaa):
        if i < 2:
            new_individual = random_cycle(distanceMatrix)
        elif i >= 2 and i<4:
            new_individual = random_cycle_inverse(distanceMatrix)[::-1]
        elif i >= 4 and i < 5:
            new_individual = generate_individual_greedy(distanceMatrix, numCities)
        elif i >= 5 and i < 7:
            new_individual = generate_individual_greedy_inverse(numCities, distanceMatrix)
        elif i >= 7 and i < lambdaa * 0.3:
            new_individual = generate_individual_nearest_neighbor(numCities, distanceMatrix)
        elif i >= lambdaa * 0.3 and i < lambdaa * 0.4:
            new_individual = generate_individual_nearest_neighbor_indices(distanceMatrix, numCities, 2)
        else:
            new_individual = generate_individual_random(numCities, distanceMatrix)
        if not tourIsCorrect(distanceMatrix, new_individual):
            raise ValueError("Invalid tour during initialization")

        # Evaluate the individual with the objective function
        obj = fitness(distanceMatrix, new_individual)

        max_tries = numCities
        # if obj of the individual is not inf it will skip this
        while obj == np.inf and max_tries > 0:
            new_individual = generate_individual_random(numCities, distanceMatrix)
            obj = fitness(distanceMatrix, new_individual)
            max_tries -= 1
        population[i] = Individual(numCities, new_individual)
    return population

@jit(nopython=True)
def compute_variance(population):
    num_individuals = len(population)
    total_distance = 0
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            distance = np.linalg.norm(population[i].path - population[j].path)
            total_distance += distance
    return total_distance / (num_individuals * (num_individuals - 1) / 2)

# no numba
def local_search_subset(distanceMatrix, population, k):
    # Apply the best subset solution to each row of the population array
    for i in range(len(population)):
        population[i].path = improve_subset_permutations(distanceMatrix, population[i].path, k)
    return population

# no numba
def local_search_shuffle_subset(distanceMatrix, population, k):
    # Apply the best subset solution to each row of the population array
    for i in range(len(population)):
        population[i].path = improve_subset_shuffle(distanceMatrix, population[i].path, k)
    return population

""" Perform k-tournament selection to select pairs of parents. """
# @jit(nopython=True)
def selection_kTour(numCities, distanceMatrix, population, k, mu):
    # prepare future selected population
    # selected = np.zeros((mu, numCities - 1)).astype(np.int16)
    selected = np.empty(mu, dtype=object)

    for i in range(mu):
        # k-tournaments: randomly select k individuals
        # randIndices = random.choices(range(np.size(population, 0)), k = k)
        randIndices = np.random.choice(np.arange(1, population.shape[0] - 1), k, replace=False)
        # look for the best individual between the k randomly selected
        best = np.argmin(pop_fitness(distanceMatrix, population[randIndices]))

        selected[i] = Individual(numCities, population[randIndices[best]].path)

        if not tourIsCorrect(distanceMatrix, selected[i].path):
            raise ValueError("Invalid tour during selection")
    return selected

@jit(nopython=True)
def selection_topK(distanceMatrix, numCities, population, k, mu):
    # prepare future selected population
    selected = np.zeros((mu, numCities - 1)).astype(np.int16)

    # top best selection
    fvals = pop_fitness(distanceMatrix, population)
    # Sort the individuals based on their objective function value (list of ordered indices)
    sorted = np.argsort(fvals)
    for i in range(mu):
        # limit the population P to the k candidate solutions with the best objective value 
        # and sample an element from this reduced (multi)set
        sampled_number = np.random.choice(sorted[:int(k)])
        selected[i, :] = population[sampled_number, :]
        if not tourIsCorrect(distanceMatrix, selected[i, :]):
            raise ValueError("Invalid tour during selection")
    return selected

""" Perform PMX-crossover using all of the given paths"""
@jit(nopython=True)
def pmx_crossover_j(distanceMatrix, numCities, paths):
    # We will create 2 offspring for each 2 parents
    offspring = np.zeros((len(paths), numCities - 1)).astype(np.int16)
    # Take paths of even and odd indices together to create pairs of parents
    parents = list(zip(paths[::2], paths[1::2]))
    # Cross p0 with p1 and vice versa
    for i, p in enumerate(parents):
        offspring[2*i] = pmx_crossover_parents(distanceMatrix, p[0], p[1])
        offspring[2*i + 1] = pmx_crossover_parents(distanceMatrix, p[1], p[0])
    return offspring

@jit(nopython=True)
def pmx_crossover_parents(distanceMatrix, p1, p2):
    # Choose random crossover points & assign corresponding segments
    cp1, cp2 = sorted(random.sample(range(len(p1)), 2))
    seg_p1, seg_p2 = p1[cp1:cp2 + 1], p2[cp1:cp2 + 1]
    # Initialize child with segment from p1
    child = np.zeros(len(p1))
    child[cp1:cp2 + 1] = seg_p1
    # seg_idx is index of value i in seg_p2
    for seg_idx, i in enumerate(seg_p2):
        if i not in seg_p1:
            j = seg_p1[seg_idx]
            while True:
                pos_j_in_p2 = np.where(p2 == j)[0]
                if pos_j_in_p2[0] < cp1 or cp2 < pos_j_in_p2[0]:
                    child[pos_j_in_p2] = i
                    break
                else:
                    j = p1[pos_j_in_p2[0]]
    # Copy over the rest of p2 into the child
    child[child == 0] = p2[child == 0]
    if not tourIsCorrect(distanceMatrix, child):
        raise ValueError("Invalid tour during mutation")
    return child

@jit(nopython=True)
def crossover(distanceMatrix, selected: np.ndarray, mu, lambdaa):
    # Create a matrix of offspring
    offspring = np.zeros((mu, distanceMatrix.shape[0] - 1))
    offspring = offspring.astype(np.int64)

    for ii in range(lambdaa):
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambdaa), 2, replace=False))
        # Perform crossover
        offspring[ii, :], offspring[ii + lambdaa, :] = pmx(distanceMatrix, selected[ri[0], :], selected[ri[1], :])
    return offspring

@jit(nopython=True)
def pmx(distanceMatrix, parent1, parent2):
    # Create a child
    child1 = np.ones(distanceMatrix.shape[0]-1).astype(np.int64)
    child2 = np.ones(distanceMatrix.shape[0]-1).astype(np.int64)

    child1 = child1 * -1
    child2 = child2 * -1

    # select random start and end indices for parent1's subsection
    start, end = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False)) # why 1, ..?

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    child1[child1 == -1] = [i for i in parent2 if i not in child1]
    child2[child2 == -1] = [i for i in parent1 if i not in child2]

    if not tourIsCorrect(distanceMatrix, child1):
        raise ValueError("Invalid tour during crossover")
    if not tourIsCorrect(distanceMatrix, child2):
        raise ValueError("Invalid tour during crossover")
    return child1, child2

# ALPHA UPDATE
# no numba (random.sample)
def pmx_crossover(numCities, distanceMatrix, selected, k, lambdaa):
    offspring = np.empty(lambdaa, dtype=object)

    parents = list(zip(selected[::2], selected[1::2]))    
    for p in range(len(parents)):
        p1, p2 = parents[p][0].path, parents[p][1].path
        a1, a2 = parents[p][0].alpha, parents[p][1].alpha
        cp1, cp2 = sorted(random.sample(range(len(p1)), 2))  # random crossover points
        seg_p1, seg_p2 = p1[cp1:cp2 + 1], p2[cp1:cp2 + 1]
        child = np.zeros(len(p1))
        child[cp1:cp2 + 1] = seg_p1                          # copy randomly selected segment from first parent
        for ii in range(len(seg_p2)):
            i = k = seg_p2[ii]
            if i not in seg_p1:
                j = seg_p1[ii]
                assigned = False
                while not assigned:                        
                    ind_j2 = np.where(p2 == j)[0]
                    if ind_j2[0] < cp1 or cp2 < ind_j2[0]:
                        child[ind_j2] = i
                        assigned = True
                    else:
                        k = p2[ind_j2[0]]                  
                        j = p1[ind_j2[0]]                   
        child[child == 0] = p2[child == 0]

        beta = 2 * random.random() - 0.5 # Number between -0.5 and 3.5
        a = a1 + beta*(a1 - a2)
        offspring[p] = Individual(numCities, child, alpha=a)
        if not tourIsCorrect(distanceMatrix, offspring[p].path):
            raise ValueError("Invalid tour during mutation")
    return offspring

# @jit(nopython=True)
def crossover_scx(distanceMatrix, selected: np.ndarray, lambdaa):
    # Create a matrix of offspring
    offspring = np.empty(lambdaa, dtype=object)

    for ii in range(lambdaa):   
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambdaa), 2, replace=False))
        # Perform crossover using SCX
        offspring[ii] = scx(distanceMatrix, selected[ri[0]].path, selected[ri[1]].path)
    return offspring

# @jit(nopython=True)
def scx(distanceMatrix, parent1, parent2):
    # Create a child
    child = np.ones(distanceMatrix.shape[0] - 1).astype(np.int64)
    child = child * -1

    # Randomly select the starting city
    start_city = np.random.choice(np.arange(1, distanceMatrix.shape[0]))

    # Set the first city in the child
    child[0] = start_city
    not_visited = np.arange(1, distanceMatrix.shape[0])
    not_visited = np.setdiff1d(not_visited, [child[0]])

    for i in range(1, len(child)):
        # Find the index of the current city in both parents
        index_parent1 = np.where(parent1 == start_city)[0][0]
        index_parent2 = np.where(parent2 == start_city)[0][0]

        # Find the neighbors of the current city in both parents
        neighbors_parent1 = np.array([parent1[(index_parent1 - 1) % len(parent1)], parent1[(index_parent1 + 1) % len(parent1)]])
        neighbors_parent2 = np.array([parent2[(index_parent2 - 1) % len(parent2)], parent2[(index_parent2 + 1) % len(parent2)]])

        # Find the common neighbors
        common_neighbors = np.intersect1d(neighbors_parent1, neighbors_parent2)
        common_neighbors_valid = np.intersect1d(common_neighbors, not_visited)
        # If there are common neighbors, choose one randomly
        if len(common_neighbors_valid) > 0:
            next_city = np.random.choice(common_neighbors_valid)
        else:
            distances = distanceMatrix[start_city, not_visited]
            next_city_index = np.argmin(distances)   
            next_city = not_visited[next_city_index]
        # Set the next city in the child
        child[i] = next_city
        start_city = next_city
        not_visited = np.setdiff1d(not_visited, [child[i]])

    if not tourIsCorrect(distanceMatrix, child):
        raise ValueError("Invalid tour during crossover")
    return child

def mutate(distanceMatrix, numCities, offspring: np.ndarray, mutationratios: list):
    for i in range(len(offspring)):
        if random.random() < offspring[i].alpha:        
            
            mutation_operator = np.random.choice([mutation_inversion,
                                                mutation_swap,
                                                mutation_scramble, 
                                                mutation_insert], 
                                                p=mutationratios)
            offspring[i].path = mutation_operator(distanceMatrix, numCities, offspring[i].path)
        if not tourIsCorrect(distanceMatrix, offspring[i].path):
            raise ValueError("Invalid tour during mutation")
    return offspring

@jit(nopython=True)
def mutation_swap(distanceMatrix, numCities, path):
    ri = sorted(np.random.choice(np.arange(1, numCities-1), 2, replace=False))
    path[ri[0]], path[ri[1]] = path[ri[1]], path[ri[0]]
    return path

@jit(nopython=True)
def mutation_insert(distanceMatrix, numCities, path):
    ri = sorted(np.random.choice(np.arange(1, numCities-1), 2, replace=False))
    removed = path[ri[1]]
    path = np.delete(path, ri[1])
    path = np.concatenate((path[:ri[0]], np.array([removed]), path[ri[0]:]))
    return path

@jit(nopython=True)
def mutation_scramble(distanceMatrix, numCities, path):
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))
    np.random.shuffle(path[ri[0]:ri[1]])
    return path

# @jit(nopython=True)
def mutation_inversion(distanceMatrix, numCities, path):
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))
    path[ri[0]:ri[1]] = path[ri[0]:ri[1]][::-1]
    return path   

# no numba
def elimination_localSearch(numCities, distanceMatrix, joinedPopulation, countSameBestSol, i, tuning, lambdaa, mu, alpha):

    # Keep some best before local search
    fvals = pop_fitness(distanceMatrix, joinedPopulation)
    perm = np.argsort(fvals)
    n_best = int(lambdaa/3)

    # Local search on all individual
    for j in range(n_best, mu):
        new_ind = local_search_operator_2_opt(distanceMatrix, joinedPopulation[j].path)
        if new_ind is not None:
            joinedPopulation[j] = Individual(numCities, new_ind)
            if not tourIsCorrect(distanceMatrix, new_ind):
                raise ValueError("Invalid tour during 2-opt")

    best_selected = joinedPopulation[perm[0:n_best]]  # array of Individual
    best_paths = np.array([ind.path for ind in best_selected]) # array of paths
    best_selected_unique = np.unique(best_paths, axis=0) # array of paths
    best_selected_unique_ind = np.empty(len(best_selected_unique), dtype=object)
    for i in range(best_selected_unique.shape[0]):
        best_selected_unique_ind[i] = Individual(numCities, path=best_selected_unique[i]) # array of Individual

    # Select randomly the rest individuals
    random_survivors = joinedPopulation[np.random.choice(perm[n_best:], lambdaa - n_best, replace=False)]
    random_survivors_paths = np.array([ind.path for ind in random_survivors])
    random_survivors_unique = np.unique(random_survivors_paths, axis=0) # array of paths
    random_survivors_unique_ind = np.empty(len(random_survivors_unique), dtype=object)
    for i in range(len(random_survivors_unique)):
        random_survivors_unique_ind[i] = Individual(numCities, random_survivors_unique[i]) # array of Individual

    # fill missing places in the population with random 
    generate_survivors = np.empty(lambdaa - len(best_selected_unique) - len(random_survivors_unique), dtype=object)
    for i in range(len(generate_survivors)):
        generate_survivors[i] = Individual(numCities, generate_individual_random(numCities, distanceMatrix), alpha=np.random.normal(alpha, 0.03))

    not_best = np.concatenate((random_survivors_unique_ind, generate_survivors)) # array of Individual
    # apply local search to the random_survivors (which are not the best individuals)
    if countSameBestSol == 10:
        tuning += 3
    if i % 3 == 0:
        not_best = one_opt(distanceMatrix, numCities, not_best, tuning*2)        
    elif i %3 == 1:
        if tuning >= 8:
            tuning = 3
        not_best = local_search_subset(distanceMatrix, not_best, tuning)
    else:
        if tuning >= 8:
            tuning = 3
        not_best = local_search_shuffle_subset(distanceMatrix, not_best, tuning)
    
    for jj in range(len(random_survivors)):
        random_survivors[jj].alpha = random_survivors[jj].change_alpha()
        
    # Concatenate the best and random survivors
    survivors = np.concatenate((best_selected_unique_ind, not_best))
    return survivors

@jit(nopython=True)
def build_cumulatives(distanceMatrix, order: np.ndarray, length: int):
    order = order.astype(np.int16)
    cum_from_0_to_first = np.zeros((length))
    cum_from_second_to_end = np.zeros((length))

    cum_from_second_to_end[length-1-1] = distanceMatrix[order[length-1-1], 0]
    cum_from_0_to_first[0] = distanceMatrix[0, order[0]]

    for i in range(1, length - 1):
        cum_from_0_to_first[i] = cum_from_0_to_first[i - 1] \
            + distanceMatrix[order[i-1], order[i]]
        cum_from_second_to_end[length-1-i] = cum_from_second_to_end[length - i] \
            + distanceMatrix[order[length-1-i], order[length-i]]
    
    return cum_from_0_to_first, cum_from_second_to_end

@jit(nopython=True)
def local_search_operator_2_opt(distanceMatrix, order: np.ndarray):
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_fitness = fitness(distanceMatrix, order)
    length = len(order)
    best_combination = (0, 0)
    order = order.astype(np.int16)

    cum_from_0_to_first, cum_from_second_to_end = build_cumulatives(distanceMatrix, order, length)
    if cum_from_second_to_end[-1] > np.inf:
        return None

    for first in range(1, length - 2):
        fit_first_part = cum_from_0_to_first[first-1]
        if fit_first_part >= np.inf or fit_first_part > best_fitness:
            break
        fit_middle_part = 0.0
        for second in range(first + 2, length):
            fit_middle_part += distanceMatrix[order[second-1], order[second-2]]
            if fit_middle_part >= np.inf:
                break
            
            fit_last_part = cum_from_second_to_end[second]
            if fit_last_part >= np.inf:
                continue

            bridge_first = distanceMatrix[order[first-1], order[second-1]]
            bridge_second = distanceMatrix[order[first], order[second]]
            temp = fit_first_part + fit_middle_part
            if temp > best_fitness:
                continue
            
            new_fitness = temp + fit_last_part + bridge_first + bridge_second
            if new_fitness < best_fitness:
                best_combination = (first, second)
                best_fitness = new_fitness
    best_first, best_second = best_combination
    if best_first == 0: # Initial individual was best
        return None
    new_order = np.copy(order)
    new_order[best_first:best_second] = new_order[best_first:best_second][::-1]
    return new_order

# no numba
def elimination_withoutLocal(distanceMatrix, joinedPopulation: np.ndarray, lambdaa):
    # Apply the objective function to each row of the joinedPopulation array
    fvals = pop_fitness(distanceMatrix, joinedPopulation)

    # Sort the individuals based on their objective function value
    perm = np.argsort(fvals)

    # Select the best lambdaa 1/2,3 individuals
    n_best = int(lambdaa/3)
    best_survivors = joinedPopulation[perm[0 : n_best], :]

    # Select randomly the rest individuals
    random_survivors = joinedPopulation[np.random.choice(perm[n_best:], lambdaa - n_best, replace=False), :]

    # Concatenate the best and random survivors
    survivors = np.vstack((best_survivors, random_survivors))
    return survivors

# no numba
def random_cycle(distanceMatrix, goal=0, frontier=None, expanded=None):
    path_len = distanceMatrix.shape[0]
    nodes = np.arange(path_len)
    if (frontier is None) and (expanded is None):
        # frontier is a stack implemented as a list, using pop() and append().
        # The items of the stack are (node, goal-to-node-path) tuples.
        frontier = [(goal, [goal])]
        expanded = set()
    while frontier:
        u, path = frontier.pop()
        if u == goal:
            # found a cycle, but it has to contail all the nodes
            # path_len + 1, because the goal will be added also at the end
            if len(path) == path_len + 1:
                return np.array(path[1:-1])
        if u in expanded:
            continue
        expanded.add(u)
        # loop through the neighbours at a random order, to result to order in the frontier
        np.random.shuffle(nodes)
        for v in nodes:
            if (v != u) and (distanceMatrix[u][v] != np.inf):
                # this is a neighbour
                frontier.append((v, path + [v]))
    # in case it got to a dead end, rerun
    return random_cycle(distanceMatrix)

# no numba
def random_cycle_inverse(distanceMatrix, goal=0, frontier=None, expanded=None):
    path_len = distanceMatrix.shape[0]
    nodes = np.arange(path_len)
    if (frontier is None) and (expanded is None):
        # frontier is a stack implemented as a list, using pop() and append().
        # The items of the stack are (node, goal-to-node-path) tuples.
        frontier = [(goal, [goal])]
        expanded = set()
    while frontier:
        u, path = frontier.pop()
        if u == goal:
            # found a cycle, but it has to contail all the nodes
            # path_len + 1, because the goal will be added also at the end
            if len(path) == path_len + 1:
                return np.array(path[1:-1])
        if u in expanded:
            continue
        expanded.add(u)
        # loop through the neighbours at a random order, to result to order in the frontier
        np.random.shuffle(nodes)
        for v in nodes:
            if (v != u) and (distanceMatrix[v][u] != np.inf):
                # this is a neighbour
                frontier.append((v, path + [v]))
    # in case it got to a dead end, return
    return random_cycle(distanceMatrix)

# no numba
def improve_subset_permutations(distanceMatrix, tour: np.ndarray, k: int):
    tour = tour.astype(np.int64)

    # Generate one randdom index
    ri = np.random.choice(np.arange(1, distanceMatrix.shape[0] - 2 - k), 1, replace=False)[0]
    # Find the best subset solution from ri to ri+k using brute force

    # Initialize the best tour and objective function value
    best_tour = tour.copy()
    best_obj = objf_path(distanceMatrix, tour[ri:ri+k+1])

    # Generate all the possible permutations of the cities from ri to ri+k
    permutations = np.array(list(itertools.permutations(tour[ri:ri+k])))
    
    # Add tour[ri-1] and tour[ri+k] to the permutations
    permutations = np.concatenate((np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri-1], permutations), axis=1)
    permutations = np.concatenate((permutations, np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri+k]), axis=1)

    # Evaluate the objective function for each permutation
    objs = objf_permutation(distanceMatrix, permutations)
    best = np.argmin(objs)

    # Check if the best permutation is better than the original tour
    if objs[best] < best_obj:
        # Update the best tour and objective function value
        best_tour[ri:ri+k] = permutations[best, 1:-1]
    return best_tour

# no numba
def improve_subset_shuffle(distanceMatrix, tour: np.ndarray, k: int):
    tour = tour.astype(np.int64)

    # Generate one randdom index
    ri = np.random.choice(np.arange(1, distanceMatrix.shape[0] - 2 - k), 1, replace=False)[0]

    # Initialize the best tour and objective function value
    old_tour = tour.copy()
    old_fit = objf_path(distanceMatrix, tour[ri:ri+k+1])
    subset = old_tour[ri:ri+k+1]
    if old_fit == np.inf:
        # Random shuffle the cities in the window
        random.shuffle(subset)
        new_fit = objf_path(distanceMatrix, subset)
        it=0
        while new_fit < old_fit or it < math.factorial(len(subset)):
            it+=1
            random.shuffle(subset)
            new_fit = objf_path(distanceMatrix, subset)
        old_tour[ri:ri+k+1] = subset

    return old_tour

@jit(nopython=True)
def objf_path(distanceMatrix, path: np.ndarray):
    # Convert float64 indices to integers
    path = path.astype(np.int64)

    # Apply the objective function to each row of the cities array
    sum_distance = 0
    for i in range(path.shape[0] - 1):
        # Sum the distances between the cities
        sum_distance += distanceMatrix[path[i], path[i + 1]]
    return sum_distance

@jit(nopython=True)
def objf_permutation(distanceMatrix, permutations: np.ndarray):
    # Apply the objective function to each row of the permutations array
    obj = np.zeros(permutations.shape[0])

    for i in range(permutations.shape[0]):
        obj[i] = objf_path(distanceMatrix, permutations[i, :])
    return obj

@jit(nopython=True)
def generate_individual_greedy(distanceMatrix, numCities):
    # Create an individual choosing always the nearest city

    individual = np.zeros(numCities-1).astype(np.int64)
    not_visited = np.arange(1, numCities)
    nearest_city = np.argmin(distanceMatrix[0, not_visited])
    individual[0] = not_visited[nearest_city]
    not_visited = np.delete(not_visited, nearest_city)

    for ii in range(1, numCities-1):
        # Select the nearest city
        nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual  
        individual[ii] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)
    return individual

@jit(nopython=True)
def generate_individual_greedy_inverse(numCities, distanceMatrix):
    individual = np.zeros(numCities-1).astype(np.int64)
    not_visited = np.arange(1, numCities)

    nearest_city = np.argmin(distanceMatrix[not_visited, 0])
    individual[0] = not_visited[nearest_city]
    not_visited = np.delete(not_visited, nearest_city)

    for ii in range(1, numCities-1):
        nearest_city = np.argmin(distanceMatrix[not_visited,individual[ii - 1]])
        individual[ii] = not_visited[nearest_city]
        not_visited = np.delete(not_visited, nearest_city)
    return individual[::-1]

@jit(nopython=True)
def generate_individual_random(numCities, distanceMatrix):
    r = np.zeros(numCities-1).astype(np.int64)
    r = np.random.permutation(np.arange(1, distanceMatrix.shape[0])).astype(np.int64)
    return r

def generate_individual_nearest_neighbor(numCities, distanceMatrix):
    # Create an individual choosing always the nearest city , second city is random
    individual = np.zeros(numCities-1).astype(np.int64)
    not_visited = np.arange(1,numCities)

    # first city chosen randomly
    individual[0] = np.random.randint(1, numCities) 
    not_visited = np.delete(not_visited, np.where(not_visited == individual[0])[0][0])

    for k in range(1, numCities-1): 
        nearest_city = np.argmin(distanceMatrix[individual[k-1], not_visited])
        individual[k] = not_visited[nearest_city]
        not_visited = np.delete(not_visited, nearest_city)
    return individual

@jit(nopython=True)
def generate_individual_nearest_neighbor_indices(distanceMatrix, numCities, k):

    k = (distanceMatrix.shape[0] + 1)//k
    # Create an individual choosing always the nearest city, 'num' indices are chosen random
    individual = np.zeros(numCities-1).astype(np.int64)

    # Second city is random from the city accessible from the first city
    accessible_cities = np.argsort(distanceMatrix[0, :]) # index of cities sorted from smaller to bigger
    individual[0] = accessible_cities[np.random.randint(1, accessible_cities.shape[0])]
    #TODO check perchè sortare se tanto è random?

    not_visited = np.arange(1, numCities)
    not_visited = np.delete(not_visited, np.where(not_visited == individual[0])[0][0])

    for ii in range(2, numCities):
        if ii % k == 0:
            # Select randomly a city from the not visited list
            # nearest_city = random.choice(not_visited)
            nearest_city = np.random.randint(0, not_visited.shape[0])
        else:
            # Select the nearest city
            nearest_city = np.argmin(distanceMatrix[individual[ii - 1], not_visited])
        # Add the nearest city to the individual
        individual[ii-1] = not_visited[nearest_city]
        # Remove the nearest city from the not visited list
        not_visited = np.delete(not_visited, nearest_city)
    
    return individual

@jit(nopython=True)
def tourIsCorrect(distanceMatrix, tour):
    # Check if the tour is correct
    tour = tour.astype(np.int64)
    if len(np.unique(tour[:distanceMatrix.shape[0]])) != distanceMatrix.shape[0]-1:
        return False
    else:
        return True

# @jit(nopython=True)
def one_opt(distanceMatrix, numCities, population, k):
    if k < 1:
        raise ValueError("k must be between 2 and n-1")
    for i in range(len(population)):
        # For each individual in the population
        best_tour = population[i].path
        best_obj = fitness(distanceMatrix, best_tour)

        # Select k random indices
        k_list = sorted(np.random.choice(np.arange(1, numCities-1 - 2), k, replace=False)) 
        for ri in k_list:
            # Swap the ri-th and ri+1-th cities
            tour = population[i].path.copy()
            tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

            # Evaluate the new tour
            new_obj = fitness(distanceMatrix, tour)

            # Check if the new tour is better
            if new_obj < best_obj:
                best_tour = tour
                best_obj = new_obj
        population[i] = Individual(numCities, best_tour)
    return population

""" Compute the objective function of a population of individuals"""
# @jit(nopython=True)
def pop_fitness(distanceMatrix, population: np.ndarray):
    return np.array([fitness(distanceMatrix, p.path) for p in population])

""" Compute the objective function of a candidate"""
@jit(nopython=True)
def fitness(distanceMatrix, path):
    path = path.astype(np.int16)
    sum = distanceMatrix[0, path[0]] + distanceMatrix[path[len(path) - 1], 0]
    for i in range(len(path)-1):                                                  
        sum += distanceMatrix[path[i]][path[i + 1]]
    return sum

@jit(nopython=True)
def fitness_subpath(subpath, distanceMatrix):
    subpath = subpath.astype(np.int16)
    sum = 0
    for i in range(len(subpath) - 1):
        sum += distanceMatrix[subpath[i]][subpath[i + 1]]
    return sum

@jit(nopython=True)
def plot_variance(var, best):
    plt.plot(var, label='Variance')
    plt.xlabel('Iterations')
    plt.show()

    plt.plot(best, label='best')
    plt.xlabel('Iterations')

@jit(nopython=True)
def plot_graph(mean, best, alphas):
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

    plt.plot(alphas, label='Average alphas')
    plt.title('Average alphas moving')
    plt.xlabel('Iterations')
    plt.ylabel('Alpha')
    plt.legend()
    plt.show()

def plot_hist(results_iterations):
    plt.hist(results_iterations, bins=20, color='green')
    plt.xlabel('Best Fitness')
    plt.ylabel('Count of Iterations')
    plt.title('Histogram of Best Fitness Values')
    plt.show()
    
def plot_hist_mean(mean_iterations):
    plt.hist(mean_iterations, bins=20, color='green')
    plt.xlabel('Mean Fitness')
    plt.ylabel('Count of Iterations')
    plt.title('Histogram of Mean Fitness Values')
    plt.show()
# --------------------------------------------------------- #

filename = "data/tour500.csv"
if __name__ == "__main__":
    r0978353().optimize(filename)
        

