import numpy as np
import random
import time
import Utilis
import matplotlib.pyplot as plt
import itertools 
import math


def read_from_file(filename):
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    return distanceMatrix


class TSP:
    """ Parameters """
    def __init__(self, fitness, filename):
        self.alpha = 0.23                           # Mutation probability
        self.mutationratios = [10, 1, 1, 15]        # swap, insert, scramble, inversion -> mutation ratio
        self.lambdaa = 150                          # Population size
        self.mu = self.lambdaa * 2                  # Offspring size       
        self.k = 3                                  # Tournament selection
        self.numIters = 2000                        # Maximum number of iterations
        self.objf = fitness                         # Objective function
        self.maxSameBestSol = 200        
        self.distanceMatrix = read_from_file(filename)
        self.numCities = self.distanceMatrix.shape[0]         
        self.maxTime = 300                          # Maximum 5 minutes

        # ---------------------------------------------------------------------------------------------- #

    def optimize(self):
        startTotTime = time.time()
        i=0
        mean = []
        best = []
        bestSolutions = []
        var_list = []
        
        # Termination criteria
        previousBestFitness = 0
        countSameBestSol = 0
        bestFitness = 0.0
        bestSolution = np.zeros(self.numCities - 1)

        # TODO CHECK
        tuning = 3
        var = 0

        startInitialization = time.time()
        self.population = TSP.initialization_mix(self)
        initializationTime = time.time() - startInitialization
        print("--- Initialization: %s seconds ---" % initializationTime)
        timePassed = initializationTime
    
        terminationCriteria = True    
        while terminationCriteria:
            start = time.time()
            startselection = time.time()
            selected = self.selection_kTour(self.population, self.k)                          # selected = initial*2
            # selected = self.selection_variation(self.population, self.k, var)
            # selected = self.selection_topK(self.population, len(self.population)*0.75)
            selectiontime = time.time() - startselection

            startcross = time.time()
            # offspring = self.pmx_crossover(selected, self.k)                            # offspring = initial
            # offspring = self.crossover(selected)  
            offspring = self.crossover_scx(selected)   
            # offspring = self.pmx_crossover_j(selected)            
            crossstime = time.time() - startcross

            startmutate = time.time()
            joinedPopulation = np.vstack((self.mutate(offspring), self.mutate(self.population)))   # joinedPopulation = Initial polutation + mutated children = lambdaa*2
            mutatetime = time.time() - startmutate

            startelemination = time.time()

            # self.population = self.elimination(joinedPopulation) # TODO put local search before elimination
            self.population = self.elimination_localSearch(joinedPopulation, countSameBestSol, i, tuning)
            elimtime = time.time() - startelemination

            itT = time.time() - start
            fvals = pop_fitness(self.population, self.distanceMatrix)                   # compute the fitness of the population

            # Save progress
            timePassed += (time.time() - start)
            previousBestFitness = bestFitness
            bestFitness = np.min(fvals)
            bestSolution = self.population[np.argmin(fvals), :]
            bestSolutions.append(bestSolution)
            mean.append(np.mean(fvals))
            best.append(np.min(fvals))

            var = TSP.compute_variance(self.population)
            var_list.append(var)

            print(f'{i}) {timePassed: .2f}s:\t Mean fitness = {mean[i]: .2f} \t Best fitness = {best[i]: .2f}, variance = {var: .2f}, \tpop shape = {tsp.population.shape}\t selection = {selectiontime : .2f}s, cross = {crossstime: .2f}s, mutate = {mutatetime: .2f}s, elim = {elimtime: .2f}s')
            i=i+1

            timeLeft = 300 - timePassed
            if(i % 20 == 0):
                print("---- Time left: ", int(timeLeft))
            # Termination criteria 1: time limit
            if timeLeft <= 0:
                terminationCriteria = False
                print("Terminated because of number of time limit!")
                break
            # Termination criteria 2: number of iterations
            if i >= self.numIters:
                terminationCriteria = False
                print("Terminated because of number of iteration limit!")
                break
            # Termination criteria 3: bestFitness doesn't improve for 'maxSameBestSol' times
            if bestFitness == previousBestFitness and bestFitness != np.inf:
                countSameBestSol += 1
            else:
                countSameBestSol = 0
            if countSameBestSol >= self.maxSameBestSol:
                terminationCriteria = False
                print("Terminated because of %d same best solutions!"%countSameBestSol)
                break

        print('Doneeee')
        totTime = time.time() - startTotTime
        print(f'Tot time: {totTime: .2f}s')
        return mean, best, i-1, var_list, bestSolution

    def compute_variance(population):
        num_individuals = len(population)
        total_distance = 0

        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                distance = np.linalg.norm(population[i] - population[j])
                total_distance += distance

        return total_distance / (num_individuals * (num_individuals - 1) / 2)

    def local_search_subset(self, population, k):
        # Apply the best subset solution to each row of the population array
        for ii in range(population.shape[0]):
            population[ii, :] = self.improve_subset_permutations(population[ii, :], k)
        return population
    
    def local_search_shuffle_subset(self, population, k):
        # Apply the best subset solution to each row of the population array
        for ii in range(population.shape[0]):
            population[ii, :] = self.improve_subset_shuffle(population[ii, :], k)
        return population

    def initialization_mix(self) -> np.ndarray:
        # Create a matrix of random individuals
        population = np.zeros((self.lambdaa, self.numCities - 1)) 

        for i in range(self.lambdaa):
            print("individual num: ", i)
            if i < 2:
                new_individual = self.random_cycle()
            elif i >= 2 and i<4:
                new_individual = self.random_cycle_inverse()[::-1]
            elif i >= 4 and i < 5:
                # print("\tgreedy")
                new_individual = TSP.generate_individual_greedy(self)
            elif i >= 5 and i < 7:
                # print("\tgreedy inverse")
                new_individual = TSP.generate_individual_greedy_inverse(self)
            elif i >= 7 and i < self.lambdaa * 0.2:
                # print("\tnearest neighbour")
                new_individual = TSP.generate_individual_nearest_neighbor(self)
            elif i >= self.lambdaa * 0.2 and i < self.lambdaa * 0.4:
                # print("\tnearest neighbour more random indices")
                new_individual = TSP.generate_individual_nearest_neighbor_indices(self, 2)
            else:
                # print("\tjust random")
                new_individual = TSP.generate_individual_random(self)
            
            # Evaluate the individual with the objective function
            obj = fitness(new_individual, self.distanceMatrix)

            max_tries = self.numCities
            # if obj of the previous individual is not inf it will skip this
            while obj == np.inf and max_tries > 0:
                new_individual = TSP.generate_individual_random(self)
                obj = fitness(new_individual, self.distanceMatrix)
                max_tries -= 1
            print("\t", obj)
            population[i, :] = new_individual
        return population

    def initialitazion_random(self):
        self.population = np.zeros((self.lambdaa, self.numCities - 1)).astype(int)
        for i in range(self.lambdaa):
            np.random.shuffle(self.population[i, :])
    
    # Parti da 0 e guardo quali sono i nodi che non hanno infinito e ne prendo uno a caso, per ogni nodo
    def initialitazion_RandomValid(self) -> None:
        self.population = np.zeros((self.lambdaa, self.numCities-1), dtype=int)
        for i in range(self.lambdaa):
            rIndividual = np.random.permutation(np.arange(1, self.numCities))
            obj = fitness(rIndividual, self.distanceMatrix)
            while obj == np.inf:
                rIndividual = np.random.permutation(np.arange(1, self.numCities))
                obj = fitness(rIndividual, self.distanceMatrix)
            self.population[i, :] = rIndividual.astype(int)
        return self.population

    def initialitazion_randomCycle(self):
        self.population = np.zeros((self.lambdaa, self.numCities - 1)).astype(int)
        for i in range(self.lambdaa):
            self.population[i, :] = self.random_cycle()

    """ Perform k-tournament selection to select pairs of parents. """
    def selection_kTour(self, population, k):
        # prepare future selected population
        selected = np.zeros((self.mu, self.numCities - 1)).astype(int)
        for i in range(self.mu):
            # k-tournaments: randomly select k individuals
            randIndices = random.choices(range(np.size(population,0)), k = k)
            # look for the best individual between the k randomly selected
            best = np.argmin(pop_fitness(population[randIndices, :], self.distanceMatrix))

            selected[i, :] = population[randIndices[best], :]
        return selected
    
    def selection_topK(self, population, k):
        # prepare future selected population
        selected = np.zeros((self.mu, self.numCities - 1)).astype(int)

        # top best selection
        fvals = pop_fitness(population, self.distanceMatrix)
        # Sort the individuals based on their objective function value (list of ordered indices)
        sorted = np.argsort(fvals)
        for i in range(self.mu):
            # limit the population P to the k candidate solutions with the best objective value 
            # and sample an element from this reduced (multi)set
            sampled_number = np.random.choice(sorted[:int(k)])
            selected[i, :] = population[sampled_number, :]
        return selected

    # TODO non funzia vettore con var, ma poi non so come funzionerebbe
    def selection_variation(self, population, k, var):
        if var == 0:
            print("aaa")
            return self.selection(population, k)
        else:
            # Apply the objective function to each row of the joinedPopulation array
            fvals = pop_fitness(population, self.distanceMatrix)
            combined = fvals + var

            # Check for NaN values in fvals
            if np.isnan(fvals).any():
                # Handle NaN values by setting their probabilities to zero
                fvals[np.isnan(fvals)] = 0.0

            # Your existing code
            combined = fvals + var

            sum_combined = np.sum(combined[~np.isinf(combined)])
            if sum_combined == 0:
                # Handle the case where the sum is zero (division by zero)
                probabilities = np.ones(len(population)) / len(population)
            else:
                # Your existing code
                probabilities = combined / sum_combined

            selected_indices = np.random.choice(range(len(population)), size=len(population), p=combined/np.sum(combined[~np.isinf(combined)]))
            return population[selected_indices, :]

    """ Perform recombination operators on the population 
    def crossover(self, population, k):
        offspring = np.zeros((self.lambdaa, self.numCities - 1)).astype(int)
        for i in range(self.lambdaa):
            p1 = self.selection_kTour(population, k)
            p2 = self.selection_kTour(population, k)
            subpath = Utilis.longest_common_subpath(p1, p2)
            restPath = np.setdiff1d(p1, subpath)
            np.random.shuffle(restPath)
            offspring[i, :] = np.append(subpath, restPath)
        return offspring
    """
    
    """ Perform PMX-crossover using all of the given paths"""
    def pmx_crossover_j(self, paths):
        # We will create 2 offspring for each 2 parents
        offspring = np.zeros((len(paths), self.numCities - 1)).astype(int)
        # Take paths of even and odd indices together to create pairs of parents
        parents = list(zip(paths[::2], paths[1::2]))
        # Cross p0 with p1 and vice versa
        for i, p in enumerate(parents):
            offspring[2*i] = self.pmx_crossover_parents(p[0], p[1])
            offspring[2*i + 1] = self.pmx_crossover_parents(p[1], p[0])
        return offspring
    def pmx_crossover_parents(self, p1, p2):
        print(p1)
        print(p2)
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

    def crossover(self, selected: np.ndarray):
        # Create a matrix of offspring
        offspring = np.zeros((self.mu, self.distanceMatrix.shape[0] - 1))
        offspring = offspring.astype(np.int64)

        for ii in range(self.lambdaa):
            # Select two random parents
            ri = sorted(np.random.choice(np.arange(1, self.lambdaa), 2, replace=False))
            # Perform crossover
            offspring[ii, :], offspring[ii + self.lambdaa, :] = self.pmx(selected[ri[0], :], selected[ri[1], :])
        return offspring

    def pmx(self, parent1, parent2):
        # Create a child
        child1 = np.ones(self.distanceMatrix.shape[0]-1).astype(np.int64)
        child2 = np.ones(self.distanceMatrix.shape[0]-1).astype(np.int64)

        child1 = child1 * -1
        child2 = child2 * -1

        # select random start and end indices for parent1's subsection
        start, end = sorted(np.random.choice(np.arange(1, self.distanceMatrix.shape[0]), 2, replace=False)) # why 1, ..?

        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        child1[child1 == -1] = [i for i in parent2 if i not in child1]
        child2[child2 == -1] = [i for i in parent1 if i not in child2]

        if not self.tourIsValid(child1):
            print("start, end: ", start, end)
            print("parent1: ", parent1)
            print("parent2: ", parent2)
            print("child1: ", child1)
            raise ValueError("Invalid tour during crossover")
        
        if not self.tourIsValid(child2):
            print("start, end: ", start, end)
            print("parent1: ", parent1)
            print("parent2: ", parent2)
            print("child2: ", child2)
            raise ValueError("Invalid tour during crossover")
        return child1, child2

    # individual one:
    def pmx_crossover(self, selected, k):
        offspring = np.zeros((self.lambdaa, self.numCities - 1)).astype(int)
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

    def crossover_scx(self, selected: np.ndarray):
        # Create a matrix of offspring
        offspring = np.zeros((self.mu, self.distanceMatrix.shape[0] - 1))
        offspring = offspring.astype(np.int64)

        for ii in range(self.lambdaa):
            # Select two random parents
            ri = sorted(np.random.choice(np.arange(1, self.lambdaa), 2, replace=False))
            # Perform crossover using SCX
            offspring[ii, :] = self.scx(selected[ri[0], :], selected[ri[1], :])
        return offspring
    
    def scx(self, parent1, parent2):
        print(parent1)
        print(parent2)
        # Create a child
        child = np.ones(self.distanceMatrix.shape[0] - 1).astype(np.int64)
        child = child * -1

        # Randomly select the starting city
        start_city = np.random.choice(np.arange(1, self.distanceMatrix.shape[0]))

        # Set the first city in the child
        child[0] = start_city
        not_visited = np.arange(1, self.distanceMatrix.shape[0])
        not_visited = np.setdiff1d(not_visited, [child[0]])

        for i in range(1, len(child)):
            # print(child)
            # print("not vis: ", not_visited)

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
                distances = self.distanceMatrix[start_city, not_visited]
                next_city_index = np.argmin(distances)   
                next_city = not_visited[next_city_index]
            # Set the next city in the child
            child[i] = next_city
            start_city = next_city
            not_visited = np.setdiff1d(not_visited, [child[i]])

        if not self.tourIsValid(child):
            print("parent1: ", parent1)
            print("parent2: ", parent2)
            print("child: ", child)
            raise ValueError("Invalid tour during crossover")
        print("childddd: ", child)
        return child

    def mutation(self, offspring, alpha):
        i = np.where(np.random.rand(np.size(offspring, 0)) <= alpha)[0]
        np.random.shuffle(offspring[i,:])
        return offspring

    def mutate(self, offspring):
        for i in range(len(offspring)):
            if random.random() <= self.alpha:   # Check if the random value is less than or equal to alpha
                mutation_type = random.choices(
                    [self.mutation_swap, self.mutation_insert, self.mutation_scramble, self.mutation_inversion],
                    self.mutationratios)[0]
                offspring[i] = mutation_type(offspring[i])
        return offspring

    def mutation_swap(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2)) 
        path[cp1], path[cp2] = path[cp2], path[cp1]
        return path

    def mutation_insert(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2))  
        np.delete(path, cp1)
        np.insert(path, cp2, path[cp1])
        return path

    def mutation_scramble(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2)) 
        subpath = path[cp1:cp2 + 1]
        random.shuffle(subpath)
        path[cp1:cp2 + 1] = subpath
        return path

    def mutation_inversion(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2))
        path[cp1:cp2 + 1] = path[cp1:cp2 + 1][::-1]
        return path   

    def elimination_localSearch(self, joinedPopulation, countSameBestSol, i, tuning):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = pop_fitness(joinedPopulation, self.distanceMatrix)

        # Sort the individuals based on their objective function value
        perm = np.argsort(fvals)

        # Select the best ... individuals
        n_best = int(self.lambdaa/2)
        best_survivors = joinedPopulation[perm[0 : n_best], :]

        # Select randomly the rest individuals
        random_survivors = joinedPopulation[np.random.choice(perm[n_best:], self.lambdaa - n_best, replace=False), :]

        # apply local search to the random_survivors (which are not the best individuals)
        if countSameBestSol == 10:
            tuning += 3
        if i % 3 == 0:
            print("one opt, k: ", tuning*2)
            random_survivors = self.one_opt(random_survivors, tuning*2)        
        elif i %3 == 1:
            print("subset, window: ", tuning)
            if tuning >= 8:
                tuning = 3
            random_survivors = self.local_search_subset(random_survivors, tuning)
        else:
            if tuning >= 8:
                tuning = 3
            print("subset, window, shuffle: ", tuning)
            random_survivors = self.local_search_shuffle_subset(random_survivors, tuning)

        # Concatenate the best and random survivors
        survivors = np.vstack((best_survivors, random_survivors))
        return survivors

    def elimination(self, joinedPopulation: np.ndarray):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = pop_fitness(joinedPopulation, self.distanceMatrix)

        # Sort the individuals based on their objective function value
        perm = np.argsort(fvals)

        # Select the best lambdaa/2 individuals
        n_best = int(self.lambdaa/3)
        best_survivors = joinedPopulation[perm[0 : n_best], :]

        # Select randomly the rest individuals
        random_survivors = joinedPopulation[np.random.choice(perm[n_best:], self.lambdaa - n_best, replace=False), :]

        # Concatenate the best and random survivors
        survivors = np.vstack((best_survivors, random_survivors))
        return survivors

    def random_cycle(self, goal=0, frontier=None, expanded=None):
        path_len = self.distanceMatrix.shape[0]
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
                if (v != u) and (self.distanceMatrix[u][v] != np.inf):
                    # this is a neighbour
                    frontier.append((v, path + [v]))
        # in case it got to a dead end, rerun
        return self.random_cycle()
    
    def random_cycle_inverse(self, goal=0, frontier=None, expanded=None):
        path_len = self.distanceMatrix.shape[0]
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
                if (v != u) and (self.distanceMatrix[v][u] != np.inf):
                    # this is a neighbour
                    frontier.append((v, path + [v]))
        # in case it got to a dead end, return
        return self.random_cycle()
    
    def improve_subset_permutations(self, tour: np.ndarray, k: int):
        tour = tour.astype(np.int64)

        # Generate one randdom index
        ri = np.random.choice(np.arange(1, self.distanceMatrix.shape[0] - 2 - k), 1, replace=False)[0]
        # Find the best subset solution from ri to ri+k using brute force

        # Initialize the best tour and objective function value
        best_tour = tour.copy()
        best_obj = self.objf_path(tour[ri:ri+k+1])

        # Generate all the possible permutations of the cities from ri to ri+k
        permutations = np.array(list(itertools.permutations(tour[ri:ri+k])))
        
        # Add tour[ri-1] and tour[ri+k] to the permutations
        permutations = np.concatenate((np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri-1], permutations), axis=1)
        permutations = np.concatenate((permutations, np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri+k]), axis=1)

        # Evaluate the objective function for each permutation
        objs = self.objf_permutation(permutations)
        best = np.argmin(objs)

        # Check if the best permutation is better than the original tour
        if objs[best] < best_obj:
            # Update the best tour and objective function value
            best_tour[ri:ri+k] = permutations[best, 1:-1]
        return best_tour
    
    def improve_subset_shuffle(self, tour: np.ndarray, k: int):
        tour = tour.astype(np.int64)

        # Generate one randdom index
        ri = np.random.choice(np.arange(1, self.distanceMatrix.shape[0] - 2 - k), 1, replace=False)[0]

        # Initialize the best tour and objective function value
        old_tour = tour.copy()
        old_fit = tsp.objf_path(tour[ri:ri+k+1])
        subset = old_tour[ri:ri+k+1]
        if old_fit == np.inf:
            # print("inf")
            # Random shuffle the cities in the window
            random.shuffle(subset)
            new_fit = tsp.objf_path(subset)
            it=0
            while new_fit < old_fit or it < math.factorial(len(subset)):
                # print("still inf, now: ",subset, ", ", it)
                it+=1
                random.shuffle(subset)
                new_fit = tsp.objf_path(subset)
            old_tour[ri:ri+k+1] = subset

        return old_tour

    def objf_path(self, tour: np.ndarray):
        # Convert float64 indices to integers
        tour = tour.astype(np.int64)
    
        # Apply the objective function to each row of the cities array
        sum_distance = 0
    
        for ii in range(tour.shape[0] - 1):
            # Sum the distances between the cities
            sum_distance += self.distanceMatrix[tour[ii], tour[ii + 1]]
    
        return sum_distance

    """
    #TODO try with DFS
    def improve_subset(self, tour: np.ndarray, k: int):
        tour = tour.astype(np.int64)

        # Generate one randdom index
        ri = np.random.choice(np.arange(1, self.distanceMatrix.shape[0] - 1 - k), 1, replace=False)[0]  
        # Initialize the best tour and objective function value
        old_tour = tour.copy()
        old_obj = self.obj_path(tour[ri:ri+k+1]) # +1 to consider "the end" of the subset 
        # (which is the first element in the individual after the subset)

        # Generate all the possible permutations of the cities from ri to ri+k
        permutations = np.array(list(itertools.permutations(tour[ri:ri+k])))

        # Add tour[ri-1] and tour[ri+k] to the permutations
        permutations = np.concatenate((np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri-1], permutations), axis=1)
        permutations = np.concatenate((permutations, np.ones((permutations.shape[0], 1)).astype(np.int64) * tour[ri+k]), axis=1)

        print("permutations shape:", permutations.shape)
        print("tour[ri+k] shape:", np.ones((permutations.shape[0], 1)).astype(np.int64).shape)

        # Evaluate the objective function for each permutation
        objs = pop_fitness(permutations, self.distanceMatrix)
        print(objs.shape)
        best = np.argmin(objs)  
        # TODO prendere direttamente il min!

        # Check if the best permutation is better than the original tour
        if objs[best] < old_obj:
            # Update the best tour and objective function value
            old_tour[ri:ri+k] = permutations[best, 1:-1]
        return old_tour
    """

    def objf_permutation(self, permutations: np.ndarray):
        # Apply the objective function to each row of the permutations array
        obj = np.zeros(permutations.shape[0])
    
        for ii in range(permutations.shape[0]):
            obj[ii] = self.objf_path(permutations[ii, :])
    
        return obj

    """
    def generate_individual_greedy_new(self):
        # Create an individual choosing always the nearest city
        individual = np.zeros(self.numCities-1).astype(np.int64)
        not_visited = np.arange(1, self.numCities)

        nearest_city_rows = np.argmin(self.distanceMatrix[0, not_visited])
        nearest_city_col = np.argmin(self.distanceMatrix[not_visited, 0])

        if self.distanceMatrix[0, nearest_city_col] > self.distanceMatrix[nearest_city_rows, 0]:
            nearest = nearest_city_rows
        else:
            nearest = nearest_city_col
        individual[0] = not_visited[nearest]
        not_visited = np.delete(not_visited, nearest)

        for ii in range(1, self.numCities-1):
            # Select the nearest city
            nearest_city = np.argmin(self.distanceMatrix[individual[ii - 1], not_visited])

            nearest_city_rows = np.argmin(self.distanceMatrix[individual[ii - 1], not_visited])
            nearest_city_col = np.argmin(self.distanceMatrix[not_visited, individual[ii - 1]])
            if self.distanceMatrix[individual[ii - 1], nearest_city_col] > self.distanceMatrix[nearest_city_rows, individual[ii - 1]]:
                nearest_city = nearest_city_rows
            else:
                nearest_city = nearest_city_col

            # Add the nearest city to the individual
            individual[ii] = not_visited[nearest_city]
            # Remove the nearest city from the not visited list
            not_visited = np.delete(not_visited, nearest_city)
            # print(fitness(individual, self.distanceMatrix))
        return individual
    """

    def generate_individual_greedy(self):
        # Create an individual choosing always the nearest city

        individual = np.zeros(self.numCities-1).astype(np.int64)
        not_visited = np.arange(1, self.numCities)
        nearest_city = np.argmin(self.distanceMatrix[0, not_visited])
        individual[0] = not_visited[nearest_city]
        not_visited = np.delete(not_visited, nearest_city)

        for ii in range(1, self.numCities-1):
            # Select the nearest city
            nearest_city = np.argmin(self.distanceMatrix[individual[ii - 1], not_visited])
            # Add the nearest city to the individual  
            individual[ii] = not_visited[nearest_city]
            # Remove the nearest city from the not visited list
            not_visited = np.delete(not_visited, nearest_city)
            # print(fitness(individual, self.distanceMatrix))
        return individual
    
    def generate_individual_greedy_inverse(self):
        individual = np.zeros(self.numCities-1).astype(np.int64)
        not_visited = np.arange(1, self.numCities)

        nearest_city = np.argmin(self.distanceMatrix[not_visited, 0])
        individual[0] = not_visited[nearest_city]
        not_visited = np.delete(not_visited, nearest_city)

        for ii in range(1, self.numCities-1):
            nearest_city = np.argmin(self.distanceMatrix[not_visited,individual[ii - 1]])
            individual[ii] = not_visited[nearest_city]
            not_visited = np.delete(not_visited, nearest_city)
        return individual[::-1]

    def generate_individual_random(self):
        r = np.zeros(self.numCities-1).astype(np.int64)
        r = np.random.permutation(np.arange(1, self.distanceMatrix.shape[0])).astype(np.int64)
        return r

    def tourIsValid(self, path):
        path = path.astype(np.int64)
        if len(np.unique(path[:self.numCities])) != self.numCities-1:
            return False
        else:
            return True
    
    def generate_individual_nearest_neighbor(self):
        # Create an individual choosing always the nearest city , second city is random
        individual = np.zeros(self.numCities-1).astype(np.int64)
        not_visited = np.arange(1,self.numCities)

        # first city chosen randomly
        individual[0] = np.random.randint(1, self.numCities) 
        not_visited = np.delete(not_visited, np.where(not_visited == individual[0])[0][0])

        for k in range(1, self.numCities-1): 
            nearest_city = np.argmin(self.distanceMatrix[individual[k-1], not_visited])
            individual[k] = not_visited[nearest_city]
            not_visited = np.delete(not_visited, nearest_city)
        return individual
    
    """ # should be correct, i check and checked but then problem if index 0 in pmx
    def generate_individual_nearest_neighbor_indices(self, num):
        # Create an individual choosing always the nearest city, 'num' indices are chosen random
        individual = np.zeros(self.numCities-1).astype(np.int64)
        not_visited = np.arange(1,self.numCities)

        randomIndices = np.random.choice(range(self.numCities-1), size=num, replace=False)
        for r in randomIndices:
            individual[r] = np.random.randint(1, self.numCities) 
            not_visited = np.delete(not_visited, np.where(not_visited == individual[r]))

        # Start from the begininng fill all cities
        for k in range(self.numCities-1): 
            if k in randomIndices:
                continue
            nearest_city = np.argmin(self.distanceMatrix[individual[k-1], not_visited])
            individual[k] = not_visited[nearest_city]
            not_visited = np.delete(not_visited, nearest_city)

        return individual
    """
    
    def generate_individual_nearest_neighbor_indices(self, k):

        k = (self.distanceMatrix.shape[0] + 1)//k
        # Create an individual choosing always the nearest city, 'num' indices are chosen random
        individual = np.zeros(self.numCities-1).astype(np.int64)

        # Second city is random from the city accessible from the first city
        accessible_cities = np.argsort(self.distanceMatrix[0, :]) # index of cities sorted from smaller to bigger
        individual[0] = accessible_cities[np.random.randint(1, accessible_cities.shape[0])]
        #TODO check perchè sortare se tanto è random?

        not_visited = np.arange(1, self.numCities)
        not_visited = np.delete(not_visited, np.where(not_visited == individual[0])[0][0])

        for ii in range(2, self.numCities):
            if ii % k == 0:
                # Select randomly a city from the not visited list
                # nearest_city = random.choice(not_visited)
                nearest_city = np.random.randint(0, not_visited.shape[0])
            else:
                # Select the nearest city
                nearest_city = np.argmin(self.distanceMatrix[individual[ii - 1], not_visited])
            # Add the nearest city to the individual
            individual[ii-1] = not_visited[nearest_city]
            # Remove the nearest city from the not visited list
            not_visited = np.delete(not_visited, nearest_city)
        
        return individual

    def tourIsCorrect(self, tour):
        # Check if the tour is correct
        tour = tour.astype(np.int64)
        if len(np.unique(tour[:self.distanceMatrix.shape[0]])) != self.distanceMatrix.shape[0]-1:
            return False
        else:
            return True

    """
    # MUCH LESS EFFICIENT
    def one_opt_cc(population, k):
        new_population = np.copy(population)
        for i in range(len(new_population)):
            path = new_population[i]
            improved = True
            while improved:
                improved = False
                indices_to_flip = np.random.choice(len(path) - 1, k, replace=False)

                for index in indices_to_flip:
                    j = index
                    l = (index + 2) % (len(path) - 1)  # Ensure l is within bounds

                    # Reverse the subpath in-place
                    path[j:l] = np.flip(path[j:l])

                    if tsp.tourIsValid(path):
                        new_fitness = fitness(path, tsp.distanceMatrix)
                        if new_fitness < fitness(new_population[i], tsp.distanceMatrix):
                            improved = True
                        else:
                            # Revert the reverse if it doesn't improve the fitness
                            path[j:l] = np.flip(path[j:l])
        return new_population
    """
        
    def one_opt(self, population, k):
        if k < 1 or k > population.shape[1] - 1:
            raise ValueError("k must be between 2 and n-1")
        for i in range(population.shape[0]):
            # For each individual in the population
            best_tour = population[i, :]
            best_obj = fitness(best_tour, self.distanceMatrix)

            # Select k random indices
            k_list = sorted(np.random.choice(np.arange(1, population.shape[1] - 2), k, replace=False)) 
            for ri in k_list:
                # Swap the ri-th and ri+1-th cities
                tour = population[i, :].copy()
                tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

                # Evaluate the new tour
                new_obj = fitness(tour, self.distanceMatrix)

                # Check if the new tour is better
                if new_obj < best_obj:
                    best_tour = tour
                    best_obj = new_obj
            population[i, :] = best_tour
        return population

""" Compute the objective function of a population of individuals"""
def pop_fitness(population, distanceMatrix):
    return np.array([fitness(path, distanceMatrix) for path in population])

""" Compute the objective function of a candidate"""
def fitness(path, distanceMatrix):
    path = path.astype(int)
    sum = distanceMatrix[0, path[0]] + distanceMatrix[path[len(path) - 1], 0]
    for i in range(len(path)-1):                                                    # TODO CHECK -1 ?
        sum += distanceMatrix[path[i]][path[i + 1]]
    return sum

def fitness_subpath(subpath, distanceMatrix):
    subpath = subpath.astype(int)
    sum = 0
    for i in range(len(subpath) - 1):
        sum += distanceMatrix[subpath[i]][subpath[i + 1]]
    return sum

def plot_variance(var, best):
    plt.plot(var, label='Variance')
    plt.xlabel('Iterations')
    plt.show()

    plt.plot(best, label='best')
    plt.xlabel('Iterations')
    # plt.title('Variance and Best fitness')
    # plt.ylabel('')
    # plt.legend()


# --------------------------------------------------------- #

tsp = TSP(fitness, "tour50.csv")

mean, best, it, var, bestInd = tsp.optimize()
print(bestInd)
plot_variance(var, best)

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
    # 37600        (time 264, it 600, pop 100 alpha 0.23, var finale 115, resto bordello del ultimo commit)

# tour500: simple greedy heuristic                                                                                                  (TARGET 141k)
    # 162248 (time 79  it 344  pop 100 alpha 0.1  init mix)
    # 160392.. (time ..  it ..   pop 200 alpha 0.23  init mix)
    # 159899 (elimination half half, pop 100)
    # 154707 (elimination half, tuning lore TODO check, local search subset + one_opt, pop 200)
    # 152k  (300 iterazioni, pop 100, elimination half, pmx lore, local search tutte e 3, alpha 0.3)

# tour750: simple greedy heuristic                                                                                                  (TARGET 195k), prof 197541 
    # 204010.. (time ..  it 600   pop 200 alpha 0.23  init mix)
    # 158k, 197k         (elimination half, tuning lore TODO check, local search subset + one_opt, pop 200)
    # 195k               (uguale a sopra ma random_cycle iniziale)

# tour1000: simple greedy heuristic                                                                                                 (TARGET 193k), prof 195848
    # 209884    init 126s (senza random_cycle) (it 100, pop 300, alpha 0.23, init mix, one_opt troppo alto elim sempre 1.15s)
    # 209764    init 42s (senza random_cycle) (it 100, pop 100, alpha 0.23, init mix, one_opt più basso (5) elim 0.6s)
    # 203041    init 42s (senza random_cycle) (pop 100, alpha 0.23, init mix, elim basic )
    # 203k      in 200 iterazioni velocissimo con pop 20!!
    # 196667    in 150 iterazioni, time limit, 300 pop (non sono sicura degli altri param, sicuro random-cycle iniziale, niente nn indici)
    # 202k      