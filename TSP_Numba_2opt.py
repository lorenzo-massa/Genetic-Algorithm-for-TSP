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

class r0978353:
    """ Parameters """
    def __init__(self):
        self.reporter = Reporter.Reporter(__class__.__name__)

    def optimize(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Store information at each iteration of the main loop
        mean = []
        best = []
        bestSolutions = []
        var_list = []
        var = 0
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
        population = initialization_mix()
        initializationTime = time.time() - startInitialization
        print("--- Initialization: %s seconds ---" % initializationTime)
        timePassed = initializationTime
    
        while terminationCriteria:
            # Selection
            start = time.time()
            startselection = time.time()
            selected = selection_kTour(population, k)                                                         # selected = initial*2
            # selected = selection_variation(population, k, var)
            # selected = selection_topK(population, len(population)*0.75)
            selectiontime = time.time() - startselection

            # Recombination
            startcross = time.time()
            offspring = pmx_crossover(selected, k)                                                               # offspring = initial
            # offspring = crossover(selected)  
            # offspring = crossover_scx(selected)   
            # offspring = pmx_crossover_j(selected)            
            crossstime = time.time() - startcross

            # Mutation
            startmutate = time.time()
            joinedPopulation = np.vstack((population, offspring))                                                     # joinedPopulation = Initial polutation + mutated children = lambdaa*2
            sorted_population = joinedPopulation[np.argsort(pop_fitness(joinedPopulation))]
            # to mutate more poulation thaqn just the offspring, but preserve the best 10 solutions
            joinedPopulation = np.vstack((sorted_population[:10, :], mutate(sorted_population[10:, :], mutationratios)))
            mutatetime = time.time() - startmutate

            # Elimination and Local Search
            startelemination = time.time()
            # population = elimination_withoutLocal(joinedPopulation) # TODO put local search before elimination
            population = elimination_localSearch(joinedPopulation, countSameBestSol, i, tuning)
            elimtime = time.time() - startelemination

            # Compute fitness of this current iteration
            itT = time.time() - start
            fvals = pop_fitness(population)                   
            # Save all progressmutate
            timePassed += (time.time() - start)
            previousBestFitness = bestFitness
            bestFitness = np.min(fvals)
            bestSolution = population[np.argmin(fvals), :]
            bestSolutions.append(bestSolution)
            mean.append(np.mean(fvals))
            best.append(np.min(fvals))
            ## Compute Variance
            # var = compute_variance(population)
            # var_list.append(var)

            timeLeft = self.reporter.report(mean[i], best[i], bestSolution)
            print(f'{i}) {timePassed: .2f}s:\t Mean fitness = {mean[i]: .2f} \t Best fitness = {best[i]: .2f}, variance = {var: .2f}, \tpop shape = {population.shape}\t selection = {selectiontime : .2f}s, cross = {crossstime: .2f}s, mutate = {mutatetime: .2f}s, elim = {elimtime: .2f}s')
            i=i+1

            # Check for termination  TODO FUNCTION
            if(i % 20 == 0):
                print("---- Time left: ", int(timeLeft))
            # Termination criteria 1: time limit
            if timeLeft <= 0:
                terminationCriteria = False
                print("Terminated because of time limit!")
                break
            # Termination criteria 2: number of iterations
            if i >= numIters:
                terminationCriteria = False
                print("Terminated because of number of iterations limit!")
                break
            # Termination criteria 3: bestFitness doesn't improve for 'maxSameBestSol' times
            if bestFitness == previousBestFitness and bestFitness != np.inf:
                countSameBestSol += 1
            else:
                countSameBestSol = 0
            if countSameBestSol >= maxSameBestSol:
                terminationCriteria = False
                print("Terminated because of %d same best solutions!"%countSameBestSol)
                break

        print('Doneeeeee')
        totTime = time.time() - startTotTime
        print(f'Tot time: {totTime: .2f}s')

        return mean, best[-1], bestSolution, i-1

def read_from_file(filename):
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    return distanceMatrix

def initialization_mix() -> np.ndarray:
    # Create a matrix of random individuals
    population = np.zeros((lambdaa, numCities - 1)) 

    for i in range(lambdaa):
        print("individual num: ", i)
        if i < 2:
            new_individual = random_cycle()
        elif i >= 2 and i<4:
            new_individual = random_cycle_inverse()[::-1]
        elif i >= 4 and i < 5:
            new_individual = generate_individual_greedy()
        elif i >= 5 and i < 7:
            new_individual = generate_individual_greedy_inverse()
        elif i >= 7 and i < lambdaa * 0.2:
            new_individual = generate_individual_nearest_neighbor()
        elif i >= lambdaa * 0.2 and i < lambdaa * 0.4:
            new_individual = generate_individual_nearest_neighbor_indices( 2)
        else:
            new_individual = generate_individual_random()

        if not tourIsCorrect(new_individual):
            print("new_individual: ", new_individual)

        # Evaluate the individual with the objective function
        obj = fitness(new_individual)

        max_tries = numCities
        # if obj of the previous individual is not inf it will skip this
        while obj == np.inf and max_tries > 0:
            new_individual = generate_individual_random()
            obj = fitness(new_individual)
            max_tries -= 1
        print("\t", obj)
        population[i, :] = new_individual
    return population

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def initialitazion_random():
    population = np.zeros((lambdaa, numCities - 1)).astype(np.int16)
    for i in range(lambdaa):
        np.random.shuffle(population[i, :])


# Parti da 0 e guardo quali sono i nodi che non hanno infinito e ne prendo uno a caso, per ogni nodo
@jit(nopython=True)
def initialitazion_RandomValid() -> None:
    population = np.zeros((lambdaa, numCities-1), dtype=int)
    for i in range(lambdaa):
        rIndividual = np.random.permutation(np.arange(1, numCities))
        obj = fitness(rIndividual)
        while obj == np.inf:
            rIndividual = np.random.permutation(np.arange(1, numCities))
            obj = fitness(rIndividual)
        population[i, :] = rIndividual.astype(np.int16)
    return population

@jit(nopython=True)
def initialitazion_randomCycle():
    population = np.zeros((lambdaa, numCities - 1)).astype(np.int16)
    for i in range(lambdaa):
        population[i, :] = random_cycle()

@jit(nopython=True)
def compute_variance(population):
    num_individuals = len(population)
    total_distance = 0

    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            distance = np.linalg.norm(population[i] - population[j])
            total_distance += distance

    return total_distance / (num_individuals * (num_individuals - 1) / 2)

def local_search_subset( population, k):
    # Apply the best subset solution to each row of the population array
    for ii in range(population.shape[0]):
        population[ii, :] = improve_subset_permutations(population[ii, :], k)
    return population

def local_search_shuffle_subset( population, k):
    # Apply the best subset solution to each row of the population array
    for ii in range(population.shape[0]):
        population[ii, :] = improve_subset_shuffle(population[ii, :], k)
    return population

""" Perform k-tournament selection to select pairs of parents. """
@jit(nopython=True)
def selection_kTour( population, k):
    # prepare future selected population
    selected = np.zeros((mu, numCities - 1)).astype(np.int16)
    for i in range(mu):
        # k-tournaments: randomly select k individuals
        # randIndices = random.choices(range(np.size(population, 0)), k = k)
        randIndices = np.random.choice(np.arange(1, population.shape[0] - 1), k, replace=False)
        # look for the best individual between the k randomly selected
        best = np.argmin(pop_fitness(population[randIndices, :]))

        selected[i, :] = population[randIndices[best], :]

        if not tourIsCorrect(selected[i, :]):
            print("selected: ", selected[i, :])
            raise ValueError("Invalid tour during selection")
    return selected

@jit(nopython=True)
def selection_topK( population, k):
    # prepare future selected population
    selected = np.zeros((mu, numCities - 1)).astype(np.int16)

    # top best selection
    fvals = pop_fitness(population)
    # Sort the individuals based on their objective function value (list of ordered indices)
    sorted = np.argsort(fvals)
    for i in range(mu):
        # limit the population P to the k candidate solutions with the best objective value 
        # and sample an element from this reduced (multi)set
        sampled_number = np.random.choice(sorted[:int(k)])
        selected[i, :] = population[sampled_number, :]
    return selected

""" Perform PMX-crossover using all of the given paths"""
@jit(nopython=True)
def pmx_crossover_j( paths):
    # We will create 2 offspring for each 2 parents
    offspring = np.zeros((len(paths), numCities - 1)).astype(np.int16)
    # Take paths of even and odd indices together to create pairs of parents
    parents = list(zip(paths[::2], paths[1::2]))
    # Cross p0 with p1 and vice versa
    for i, p in enumerate(parents):
        offspring[2*i] = pmx_crossover_parents(p[0], p[1])
        offspring[2*i + 1] = pmx_crossover_parents(p[1], p[0])
    return offspring
@jit(nopython=True)
def pmx_crossover_parents( p1, p2):
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
    return child

@jit(nopython=True)
def crossover( selected: np.ndarray):
    # Create a matrix of offspring
    offspring = np.zeros((mu, distanceMatrix.shape[0] - 1))
    offspring = offspring.astype(np.int64)

    for ii in range(lambdaa):
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambdaa), 2, replace=False))
        # Perform crossover
        offspring[ii, :], offspring[ii + lambdaa, :] = pmx(selected[ri[0], :], selected[ri[1], :])
    return offspring
@jit(nopython=True)
def pmx( parent1, parent2):
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

    if not tourIsCorrect(child1):
        print("start, end: ", start, end)
        print("parent1: ", parent1)
        print("parent2: ", parent2)
        print("child1: ", child1)
        raise ValueError("Invalid tour during crossover")
    
    if not tourIsCorrect(child2):
        print("start, end: ", start, end)
        print("parent1: ", parent1)
        print("parent2: ", parent2)
        print("child2: ", child2)
        raise ValueError("Invalid tour during crossover")
    return child1, child2

# individual one:
# @jit(nopython=True) no random.sample!
def pmx_crossover( selected, k):
    offspring = np.zeros((lambdaa, numCities - 1)).astype(np.int16)
    parents = list(zip(selected[::2], selected[1::2]))      # why?
    for p in range(len(parents)):
        p1, p2 = parents[p][0], parents[p][1]
        cp1, cp2 = sorted(random.sample(range(len(p1)), 2))  # random crossover points
        seg_p1, seg_p2 = p1[cp1:cp2 + 1], p2[cp1:cp2 + 1]
        child = np.zeros(len(p1))
        child[cp1:cp2 + 1] = seg_p1                          # copy randomly selected segment from first parent
        for ii in range(len(seg_p2)):
            i = k = seg_p2[ii]
            if i not in seg_p1:
                j = seg_p1[ii]
                assigned = False
                while not assigned:                         # why?
                    ind_j2 = np.where(p2 == j)[0]
                    if ind_j2[0] < cp1 or cp2 < ind_j2[0]:
                        child[ind_j2] = i
                        assigned = True
                    else:
                        k = p2[ind_j2[0]]                   # useless?
                        j = p1[ind_j2[0]]                   # useless?
        child[child == 0] = p2[child == 0]
        offspring[p] = child
    return offspring

@jit(nopython=True)
def crossover_scx( selected: np.ndarray):
    # Create a matrix of offspring
    offspring = np.zeros((mu, distanceMatrix.shape[0] - 1))
    offspring = offspring.astype(np.int64)

    for ii in range(mu):   
        # Select two random parents
        ri = sorted(np.random.choice(np.arange(1, lambdaa), 2, replace=False))
        # Perform crossover using SCX
        offspring[ii, :] = scx(selected[ri[0], :], selected[ri[1], :])
    return offspring
@jit(nopython=True)
def scx( parent1, parent2):
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

    if not tourIsCorrect(child):
        print("parent1: ", parent1)
        print("parent2: ", parent2)
        print("child: ", child)
        raise ValueError("Invalid tour during crossover")
    return child

@jit(nopython=True)
def mutation(offspring, alpha):
    i = np.where(np.random.rand(np.size(offspring, 0)) <= alpha)[0]
    np.random.shuffle(offspring[i,:])
    return offspring

# @jit(nopython=True)
def mutate(offspring, mutationratios):
    for i in range(len(offspring)):
        if random.random() <= alpha:        # TODO SELF ADAPTIVITY FOR ALPHA
            
            mutation_operator = np.random.choice([mutation_inversion,
                                                  mutation_swap,
                                                  mutation_scramble, 
                                                  mutation_insert], 
                                                  p=mutationratios)
            
            offspring[i, :] = mutation_operator(offspring[i, :])
    return offspring

@jit(nopython=True)
def mutation_swap(path):
    ri = sorted(np.random.choice(np.arange(1, numCities-1), 2, replace=False))
    path[ri[0]], path[ri[1]] = path[ri[1]], path[ri[0]]
    return path

@jit(nopython=True)
def mutation_insert(path):
    ri = sorted(np.random.choice(np.arange(1, numCities-1), 2, replace=False))

    removed = path[ri[1]]
    path = np.delete(path, ri[1])
    path = np.concatenate((path[:ri[0]], np.array([removed]), path[ri[0]:]))
    return path

@jit(nopython=True)
def mutation_scramble( path):
    # cp1, cp2 = sorted(random.sample(range(numCities - 1), 2)) 
    # subpath = path[cp1:cp2 + 1]
    # random.shuffle(subpath)
    # path[cp1:cp2 + 1] = subpath

    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))
    np.random.shuffle(path[ri[0]:ri[1]])
    return path

@jit(nopython=True)
def mutation_inversion( path):
    # cp1, cp2 = sorted(random.sample(range(numCities - 1), 2))
    # path[cp1:cp2 + 1] = path[cp1:cp2 + 1][::-1]
    ri = sorted(np.random.choice(np.arange(1, distanceMatrix.shape[0]), 2, replace=False))
    path[ri[0]:ri[1]] = path[ri[0]:ri[1]][::-1]
    return path   

# @jit(nopython=True) because local_search_subset è senza numba
def elimination_localSearch( joinedPopulation, countSameBestSol, i, tuning):

    # keep some best before local search
    fvals = pop_fitness(joinedPopulation)
    perm = np.argsort(fvals)
    n_best = int(lambdaa/4)
    best_selected = joinedPopulation[perm[0:n_best], :]

    # Local search on all individual
    for j in range(n_best, mu):
        new_ind = local_search_operator_2_opt(joinedPopulation[j])
        if new_ind is not None:
            joinedPopulation[j] = new_ind
            if not tourIsCorrect(new_ind):
                print("NOT CORRECT: ", new_ind)
                raise ValueError("Invalid tour during 2-opt")
        
    # Select randomly the rest individuals
    random_survivors = joinedPopulation[np.random.choice(perm[n_best:], lambdaa - n_best, replace=False), :]
    # apply local search to the random_survivors (which are not the best individuals)
    if countSameBestSol == 10:
        tuning += 3
    if i % 3 == 0:
        print("one opt, k: ", tuning*2)
        random_survivors = one_opt(random_survivors, tuning*2)        
    elif i %3 == 1:
        print("subset, window: ", tuning)
        if tuning >= 8:
            tuning = 3
        random_survivors = local_search_subset(random_survivors, tuning)
    else:
        if tuning >= 8:
            tuning = 3
        print("subset, window, shuffle: ", tuning)
        random_survivors = local_search_shuffle_subset(random_survivors, tuning)

    # Concatenate the best and random survivors
    survivors = np.vstack((best_selected, random_survivors))
    return survivors


@jit(nopython=True)
def build_cumulatives(distanceMatrix: np.ndarray, order: np.ndarray, length: int):
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
def local_search_operator_2_opt(order: np.ndarray):
    """Local search operator, which makes use of 2-opt. Swap two edges within a cycle."""
    best_fitness = fitness(order)
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

def elimination_withoutLocal( joinedPopulation: np.ndarray):
    # Apply the objective function to each row of the joinedPopulation array
    fvals = pop_fitness(joinedPopulation)

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

def random_cycle(goal=0, frontier=None, expanded=None):
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
    return random_cycle()

def random_cycle_inverse( goal=0, frontier=None, expanded=None):
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
    return random_cycle()

def improve_subset_permutations( tour: np.ndarray, k: int):
    tour = tour.astype(np.int64)

    # Generate one randdom index
    ri = np.random.choice(np.arange(1, distanceMatrix.shape[0] - 2 - k), 1, replace=False)[0]
    # Find the best subset solution from ri to ri+k using brute force

    # Initialize the best tour and objective function value
    best_tour = tour.copy()
    best_obj = objf_path(tour[ri:ri+k+1])

    # Generate all the possible permutations of the cities from ri to ri+k
    permutations = np.array(list(itertools.permutations(tour[ri:ri+k])))
    
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
    return best_tour

def improve_subset_shuffle( tour: np.ndarray, k: int):
    tour = tour.astype(np.int64)

    # Generate one randdom index
    ri = np.random.choice(np.arange(1, distanceMatrix.shape[0] - 2 - k), 1, replace=False)[0]

    # Initialize the best tour and objective function value
    old_tour = tour.copy()
    old_fit = objf_path(tour[ri:ri+k+1])
    subset = old_tour[ri:ri+k+1]
    if old_fit == np.inf:
        # Random shuffle the cities in the window
        random.shuffle(subset)
        new_fit = objf_path(subset)
        it=0
        while new_fit < old_fit or it < math.factorial(len(subset)):
            it+=1
            random.shuffle(subset)
            new_fit = objf_path(subset)
        old_tour[ri:ri+k+1] = subset

    return old_tour

@jit(nopython=True)
def objf_path(path: np.ndarray):
    # Convert float64 indices to integers
    path = path.astype(np.int64)

    # Apply the objective function to each row of the cities array
    sum_distance = 0
    for i in range(path.shape[0] - 1):
        # Sum the distances between the cities
        sum_distance += distanceMatrix[path[i], path[i + 1]]
    return sum_distance

@jit(nopython=True)
def objf_permutation(permutations: np.ndarray):
    # Apply the objective function to each row of the permutations array
    obj = np.zeros(permutations.shape[0])

    for i in range(permutations.shape[0]):
        obj[i] = objf_path(permutations[i, :])
    return obj

@jit(nopython=True)
def generate_individual_greedy():
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
def generate_individual_greedy_inverse():
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
def generate_individual_random():
    r = np.zeros(numCities-1).astype(np.int64)
    r = np.random.permutation(np.arange(1, distanceMatrix.shape[0])).astype(np.int64)
    return r

@jit(nopython=True)
def generate_individual_nearest_neighbor():
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
def generate_individual_nearest_neighbor_indices(k):

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
def tourIsCorrect(tour):
    # Check if the tour is correct
    tour = tour.astype(np.int64)
    if len(np.unique(tour[:distanceMatrix.shape[0]])) != distanceMatrix.shape[0]-1:
        return False
    else:
        return True

@jit(nopython=True)  
def one_opt(population, k):
    if k < 1 or k > population.shape[1] - 1:
        raise ValueError("k must be between 2 and n-1")
    for i in range(population.shape[0]):
        # For each individual in the population
        best_tour = population[i, :]
        best_obj = fitness(best_tour)

        # Select k random indices
        k_list = sorted(np.random.choice(np.arange(1, population.shape[1] - 2), k, replace=False)) 
        for ri in k_list:
            # Swap the ri-th and ri+1-th cities
            tour = population[i, :].copy()
            tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

            # Evaluate the new tour
            new_obj = fitness(tour)

            # Check if the new tour is better
            if new_obj < best_obj:
                best_tour = tour
                best_obj = new_obj
        population[i, :] = best_tour
    return population

""" Compute the objective function of a population of individuals"""
@jit(nopython=True)
def pop_fitness(population):
    return np.array([fitness(path) for path in population])

""" Compute the objective function of a candidate"""
@jit(nopython=True)
def fitness(path):
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
    # plt.title('Variance and Best fitness')
    # plt.ylabel('')
    # plt.legend()

@jit(nopython=True)
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

# --------------------------------------------------------- #

# Parameters



f1="data/tour50.csv"
f2="data/tour100.csv"
f3="data/tour200.csv"
f4="data/tour500.csv"
f5="data/tour750.csv"
f6="data/tour1000.csv"
files = [f6]
    

iterations = range(3)
results_iterations = []
if __name__ == "__main__":
    for f in files:
        alpha = 0.5                                      # Mutation probability
        mutationratios = [0.9, 0.1, 0, 0]                # inversion, swap, scramble, insert
        lambdaa = 100                                    # Population size
        mu = lambdaa * 2                                 # Offspring size       
        k = 3                                            # Tournament selection
        numIters = 2000                                  # Maximum number of iterations
        objf = fitness                                   # Objective function
        maxSameBestSol = 100  
        maxTime = 300                                    # Maximum 5 minutes
        distanceMatrix = read_from_file(f)
        numCities = distanceMatrix.shape[0]   

        for i in iterations:
            print(f, "\t Try: ", i)
            start_time = time.time()
            mean, best_fit, best_ind, numIt = r0978353().optimize()
            end_time = time.time()
            results_iterations.append((best_fit))
    
        # Print all the results
        print("\n", str(f))
        for i, result in enumerate(results_iterations):
            print("Result ", i, ": ", result)
        print("mean of the results: ", np.mean(results_iterations))
        print("Best results found: ", np.min(results_iterations))
        print("\n")

        # # Plot histogram
        # plt.hist(results_iterations, bins=20, color='blue', edgecolor='black')
        # plt.xlabel('Best Fitness')
        # plt.ylabel('Count of Iterations')
        # plt.title('Histogram of Best Fitness Values')
        # plt.show()



# tour50: simple greedy heuristic                                                                                                   (TARGET 24k)
    # 27134 (time 170 , iter 750 , pop 500,  alpha 0.23, init mix)
    # 28971 uguale a sopra ma valori alti nella one_opt (10,15,20)
    # 26392 pmx jonas

# tour100: simple greedy heuristic 81616                                                                                            (TARGET 81k)
    # 81616 (time 78s, 1k iterations, pop 150, alpha 0.23, init randomValid)
    # 80631 (pop 500, alpha 0.23, mutate also the initial pop, init randomValid, circa 600 ite)
    # 80612 (pop 500, tutto uguale a quella sopra ma con init_mix, 709 ite)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# tour200: simple greedy heuristic                                                                                                  (TARGET 35k)
    # 48509        (time 195s, 1k iterations, pop 150, alpha 0.23, init randomValid)
    # 38514        (time   it   pop 300  alpha 0.23  init mix  one_opt alti)
    # 38325        (time 300, it 96, crossover scx lentissimo, pop 100)
    # 37600        (time 264, it 600, pop 100 alpha 0.23, var finale 115, resto bordello del ultimo commit)

# tour500: simple greedy heuristic                                                                                                  (TARGET 141k)
    # 162248    (time 79  it 344  pop 100 alpha 0.1  init mix)
    # 160392..  (time ..  it ..   pop 200 alpha 0.23  init mix)
    # 159899    (elimination half half, pop 100)
    # 159630    (time 300 it 32 elim half, (aumentata best?!), pop 100, scx crossover lentoo)
    # 154707    (elimination half, tuning lore TODO check, local search subset + one_opt, pop 200)
    # 152k      (300 iterazioni, pop 100, elimination half, pmx lore, local search tutte e 3, alpha 0.3)
    # 137k       2 opt ...sus...

# tour750: simple greedy heuristic                                                                                                  (TARGET 195k), prof 197541 
    # 204010..           (time ..  it 600   pop 200 alpha 0.23  init mix)
    # 158k, 197k         (elimination half, tuning lore TODO check, local search subset + one_opt, pop 200)
    # 198845             (time 300 it 32, elim 1/3, pop 100, (aumentata best?!), scx crossover lentoo)
    # 195k               (uguale a sopra ma random_cycle iniziale)
    # 175k                con 2-opt ...sus...

# tour1000: simple greedy heuristic                                                                                                 (TARGET 193k), prof 195848
    # 209884    init 126s (senza random_cycle) (it 100, pop 300, alpha 0.23, init mix, one_opt troppo alto elim sempre 1.15s)
    # 209764    init 42s (senza random_cycle) (it 100, pop 100, alpha 0.23, init mix, one_opt più basso (5) elim 0.6s)
    # 203041    init 42s (senza random_cycle) (pop 100, alpha 0.23, init mix, elim basic )
    # 203k      in 200 iterazioni velocissimo con pop 20!!
    # 196667    in 150 iterazioni, time limit, 300 pop (non sono sicura degli altri param, sicuro random-cycle iniziale, niente nn indici)
    # 200k      con 2-opt