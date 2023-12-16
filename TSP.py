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
        self.alpha = 0.23                           # Mutation probability
        self.mutationratios = [7, 1, 1, 15]         # swap, insert, scramble, inversion -> mutation ratio
        self.lambdaa = 1000                         # Population size
        self.mu = self.lambdaa * 2                  # Offspring size        WHY THE DOUBLE (COULD BE THE HALF?)
        self.k = 3                                  # Tournament selection
        self.numIters = 90                          # Maximum number of iterations
        self.objf = fitness                         # Objective function

        self.distanceMatrix = read_from_file(filename)
        self.numCities = self.distanceMatrix.shape[0]         # Boundary of the domain, not to be changed
        self.maxTime = 300                                    # Maximum 5 minutes

        """ Initialize the first population """
        self.population = np.zeros((self.lambdaa, self.numCities - 1)).astype(int)
        # self.population = np.vstack([np.arange(1, self.numCities)] * self.lambdaa)
        for i in range(self.lambdaa):
            self.population[i, :] = self.random_cycle()
            # np.random.shuffle(self.population[i, :])

    def optimize(self):
        startTotTime = time.time()
        i=0
        mean = []
        best = []
        while ((time.time() - startTotTime) < self.maxTime) and (i<=self.numIters):
            start = time.time()

            startselection = time.time()
            selected = self.selection(self.population, self.k)                          # selected = initial*2
            selectiontime = time.time() - startselection

            startcross = time.time()
            offspring = self.pmx_crossover(selected, self.k)                            # offspring = initial
            crossstime = time.time() - startcross

            startmutate = time.time()
            joinedPopulation = np.vstack((self.mutate(offspring), self.population))     # joinedPopulation = Initial polutation + mutated children = lambdaa*2
            mutatetime = time.time() - startmutate

            startelemination = time.time()
            self.population = self.elimination(joinedPopulation, self.lambdaa)                         # population = joinedPopulation - eliminated = lambdaa
            elimtime = time.time() - startelemination

            itT = time.time() - start
            fvals = pop_fitness(self.population, self.distanceMatrix)
            mean.append(np.mean(fvals))
            best.append(np.min(fvals))
            print(f'{i}) {itT: .2f}s:\t Mean fitness = {mean[i]: .5f} \t Best fitness = {best[i]: .5f}\t pop shape = {tsp.population.shape}\t selection = {selectiontime : .2f}s, cross = {crossstime: .2f}s, mutate = {mutatetime: .2f}s')
            i=i+1
        print('Done')
        totTime = time.time() - startTotTime
        print(f'Tot time: {totTime: .2f}s')
        return mean, best, i-1

    """ Perform k-tournament selection to select pairs of parents. """
    def selection_kTour(self, population, k):
        randIndices = random.choices(range(np.size(population,0)), k = k)
        best = np.argmin(pop_fitness(population[randIndices, :], self.distanceMatrix))
        return population[randIndices[best], :]

    def selection(self, population, k):
        selected = np.zeros((self.mu, self.numCities - 1)).astype(int)
        for i in range(self.mu):
            selected[i, :] = self.selection_kTour(population, k)
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


    """ Perform mutation, adding a random Gaussian perturbation. """
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
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2))  # Random indexes
        path[cp1], path[cp2] = path[cp2], path[cp1]
        return path

    def mutation_insert(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2))  # Random indexes
        np.delete(path, cp1)
        np.insert(path, cp2, path[cp1])
        return path

    def mutation_scramble(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2))  # Random indexes
        subpath = path[cp1:cp2 + 1]
        random.shuffle(subpath)
        path[cp1:cp2 + 1] = subpath
        return path

    def mutation_inversion(self, path):
        cp1, cp2 = sorted(random.sample(range(self.numCities - 1), 2))
        path[cp1:cp2 + 1] = path[cp1:cp2 + 1][::-1]
        return path


    """ Eliminate the unfit candidate solutions. """
    # TODO consider age-based elimination
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

def pop_fitness(population, distanceMatrix):
    return np.array([fitness(path, distanceMatrix) for path in population])

""" Compute the objective function of a candidate"""
def fitness(path, distanceMatrix):
    sum = distanceMatrix[0, path[0]] + distanceMatrix[path[len(path) - 1]][0]
    for i in range(len(path)-2):
        sum += distanceMatrix[path[i]][path[i + 1]]
    return sum

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

tsp = TSP(fitness, "../data/tour50.csv")
mean, best, it = tsp.optimize()
plot_graph(mean, best)










