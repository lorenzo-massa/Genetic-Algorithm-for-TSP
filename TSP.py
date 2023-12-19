import numpy as np
import random
import time
import Utilis
import matplotlib.pyplot as plt


def read_from_file(filename):
    # Read distance matrix from file.
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    return distanceMatrix


class TSP:
    """ Parameters """
    def __init__(self, fitness, filename):
        self.alpha = 0.1                            # Mutation probability
        self.mutationratios = [7, 1, 1, 15]         # swap, insert, scramble, inversion -> mutation ratio
        self.lambdaa = 100                          # Population size
        self.mu = self.lambdaa * 2                  # Offspring size       
        self.k = 3                                  # Tournament selection
        self.numIters = 2000                        # Maximum number of iterations
        self.objf = fitness                         # Objective function
        self.maxSameBestSol = self.numIters / 10         
        self.distanceMatrix = read_from_file(filename)
        self.numCities = self.distanceMatrix.shape[0]         
        self.maxTime = 300                          # Maximum 5 minutes

        """ Initialize the first population """
        startInitialization = time.time()
        # TSP.initialitazion_randomCycle(self)
        # TSP.initialitazion_RandomValid(self)
        TSP.initialization_mix(self)
        initializationTime = time.time() - startInitialization
        print("--- Initialization: %s seconds ---" % initializationTime)
    
    def initialization_mix(self) -> np.ndarray:
        # Create a matrix of random individuals
        self.population = np.zeros((self.lambdaa, self.numCities - 1)) 
        for i in range(self.lambdaa):
            if i < self.lambdaa*0.1:
                print("\trandom cyle")
            #   new_individual = TSP.generate_individual_greedy(self)
                new_individual = TSP.random_cycle(self)
                print("\t\t", fitness(new_individual, self.distanceMatrix))
            elif i >= self.lambdaa * 0.2 and i < self.lambdaa * 0.4:
                print("\tgreedy")
                new_individual = TSP.generate_individual_greedy(self)
                print("\t\t", fitness(new_individual, self.distanceMatrix))
            elif i >= self.lambdaa * 0.4 and i < self.lambdaa * 0.6:
                print("\tnearest neighbour")
                new_individual = TSP.generate_individual_nearest_neighbor(self)
                print("\t\t", fitness(new_individual, self.distanceMatrix))
            else:
                print("\trandom")
                new_individual = TSP.generate_individual_random(self)
                print("\t\t", fitness(new_individual, self.distanceMatrix))
            # Evaluate the individual with the objective function
            obj = fitness(new_individual, self.distanceMatrix)

            max_tries = self.numCities
            # if obj of the previous individual is not inf it will skip this
            s=""
            while obj == np.inf and max_tries > 0:
                s +="."
                new_individual = TSP.generate_individual_random(self)
                obj = fitness(new_individual, self.distanceMatrix)
                max_tries -= 1
            print(s)

            if not TSP.tourIsValid(self, new_individual):
                print("NOT VALID")
                raise ValueError("Invalid tour during initialization")
            
            # Create alpha with gaussian distribution
            # alpha = np.array([np.random.normal(self.alpha, 0.05)])
            # print(alpha)
            # Concatenate the alpha value to the individual
            # new_individual = np.concatenate((new_individual, alpha))
            self.population[i, :] = new_individual
        return self.population

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

    def optimize(self):
        startTotTime = time.time()
        i=0
        mean = []
        best = []
        bestSolutions = []
        
        # Termination criteria
        previousBestFitness = 0
        countSameBestSol = 0
        bestFitness = 0.0
        bestSolution = np.zeros(self.numCities - 1)

        terminationCriteria = True    
        while terminationCriteria:
            start = time.time()
            startselection = time.time()
            selected = self.selection(self.population, self.k)                          # selected = initial*2
            selectiontime = time.time() - startselection

            startcross = time.time()
            offspring = self.pmx_crossover(selected, self.k)                            # offspring = initial
            crossstime = time.time() - startcross

            startmutate = time.time()
            # joinedPopulation = np.vstack((self.mutate(offspring), self.population))     # joinedPopulation = Initial polutation + mutated children = lambdaa*2
            joinedPopulation = np.vstack((self.mutate(offspring), self.mutate(self.population))) 
            mutatetime = time.time() - startmutate

            startelemination = time.time()
            self.population = self.elimination(joinedPopulation, self.lambdaa)          # population = joinedPopulation - eliminated = lambdaa
            elimtime = time.time() - startelemination

            itT = time.time() - start
            fvals = pop_fitness(self.population, self.distanceMatrix)                   # compute the fitness of rhe population

            # Save progress
            previousBestFitness = bestFitness
            bestFitness = np.min(fvals)
            bestSolution = self.population[np.argmin(fvals), :]
            bestSolutions.append(bestSolution)
            mean.append(np.mean(fvals))
            best.append(np.min(fvals))

            print(f'{i}) {itT: .2f}s:\t Mean fitness = {mean[i]: .5f} \t Best fitness = {best[i]: .5f} \tpop shape = {tsp.population.shape}\t selection = {selectiontime : .2f}s, cross = {crossstime: .2f}s, mutate = {mutatetime: .2f}s, elim = {elimtime: .2f}s')
            i=i+1

            # Termination criteria 1: number of iterations
            ((time.time() - startTotTime) < self.maxTime) and (i<=self.numIters)
            if i >= self.numIters:
                terminationCriteria = False
                print("Terminated because of number of iteration limit!")
                break
            # Termination criteria 2: bestFitness doesn't improve for 'maxSameBestSol' times
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
        return mean, best, i-1

    """ Perform k-tournament selection to select pairs of parents. """
    def selection(self, population, k):
        # prepare future selected population
        selected = np.zeros((self.mu, self.numCities - 1)).astype(int)
        for i in range(self.mu):
            # k-tournaments: randomly select k individuals
            randIndices = random.choices(range(np.size(population,0)), k = k)
            # look for the best individual between the k randomly selected
            best = np.argmin(pop_fitness(population[randIndices, :], self.distanceMatrix))

            selected[i, :] = population[randIndices[best], :]

            # TODO consider this part:
            # if not self.tourIsValid(selected[i, :-1]):
            #     print("selected: ", selected[i, :-1])
            #     raise ValueError("Invalid tour during selection")
        return selected

    """ Perform recombination operators on the population """
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
    
    def elimination(self, population, lambdaa):
        fvals = pop_fitness(population, self.distanceMatrix)
        sortedFitness = np.argsort(fvals)
        return population[sortedFitness[0: lambdaa], :]     # TODO check: lambdaa - 1

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
    
    def generate_individual_greedy(self):
        # Create an individual choosing always the nearest city
        individual = np.zeros(self.numCities-1).astype(np.int64)
        not_visited = np.arange(1, self.numCities)
        for ii in range(1, self.numCities):
            # Select the nearest city
            nearest_city = np.argmin(self.distanceMatrix[individual[ii - 1], not_visited])
            # Add the nearest city to the individual
            individual[ii-1] = not_visited[nearest_city]
            # Remove the nearest city from the not visited list
            not_visited = np.delete(not_visited, nearest_city)
        return individual

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
        individual = np.zeros(self.numCities -1).astype(np.int64)
        individual[0] = np.random.randint(1, self.distanceMatrix.shape[0]) # first city is random
        not_visited = np.arange(1,self.numCities)
        not_visited = np.delete(not_visited, np.where(not_visited == individual[0])[0][0])
        for ii in range(2, self.numCities): # 2?
            nearest_city = np.argmin(self.distanceMatrix[individual[ii - 1], not_visited])
            individual[ii-1] = not_visited[nearest_city]
            not_visited = np.delete(not_visited, nearest_city)
        return individual

""" Compute the objective function of a population of individuals"""
def pop_fitness(population, distanceMatrix):
    return np.array([fitness(path, distanceMatrix) for path in population])

""" Compute the objective function of a candidate"""
def fitness(path, distanceMatrix):
    path = path.astype(int)
    sum = distanceMatrix[0, path[0]] + distanceMatrix[path[len(path) - 1]][0]
    for i in range(len(path)-1):
        sum += distanceMatrix[path[i]][path[i + 1]]
    return sum

# --------------------------------------------------------- #

tsp = TSP(fitness, "tour750.csv")

mean, best, it = tsp.optimize()
# plot_graph(mean, best)

# Here: 
# tour50: simple greedy heuristic 28772                                                                                             (TARGET 24k)
# tour100: simple greedy heuristic 81616                                                                                            (TARGET 81k)
    # 81616 (time 78s, 1k iterations, pop 150, alpha 0.23, init randomValid)
    # 80631 (pop 500, alpha 0.23, mutate also the initial pop, init randomValid, circa 600 ite)
    # 80612 (pop 500, tutto uguale a quella sopra ma con init_mix, 709 ite)
# tour200: simple greedy heuristic 48509        (time 195s, 1k iterations, pop 150, alpha 0.23, init randomValid)                   (TARGET 35k)
# tour500: simple greedy heuristic 1404427                                                                                          (TARGET 141k)
    # 938566
# tour750: simple greedy heuristic                                                                                                  (TARGET 195k)   
# tour1000: simple greedy heuristic                                                                                                 (TARGET 193k)