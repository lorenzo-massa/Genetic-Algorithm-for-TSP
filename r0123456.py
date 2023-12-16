import Reporter
import numpy as np
import random
import matplotlib.pyplot as plt
import time


# Modify the class name to match your student number.
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.

        TSP = TravelsalesmanProblem(distanceMatrix,
                                    lambda_=150,
                                    mu=300, # 2*lambda
                                    alpha=0.28,
                                    max_iterations=500)
        
        population = TSP.initialize()

        # Try to improve the initial population with local search
        population = TSP.two_opt(population, 5)
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
            if iteration < 200:
                selected = TSP.selection(population,3)
            else:
                selected = TSP.selection(population,4)

            # Crossover
            offspring = TSP.crossover(selected)
            joinedPopulation = np.vstack(
                (TSP.mutation(offspring), TSP.mutation(population))
            )

            # Elimination
            if iteration < 80:
                 population = TSP.elimination(joinedPopulation)
            elif iteration >= 80 and iteration < 150:
                joinedPopulation = TSP.two_opt(joinedPopulation, 3)
                population = TSP.elimination(joinedPopulation)
            elif iteration >= 150 and iteration < 200:
                joinedPopulation = TSP.one_opt(joinedPopulation, 3)
                population = TSP.elimination_with_crowding(joinedPopulation)
            else:
                joinedPopulation = TSP.one_opt(joinedPopulation, 9)
                population = TSP.elimination_with_crowding(joinedPopulation)


            # Show progress
            fvals = np.apply_along_axis(TSP.objf, 1, population)
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
                    "Terminated because of %d different best solutions"
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
        lambda_: int = 200,
        mu: int = 400,
        alpha: float = 0.28,
        max_iterations= 600,
    ) -> None:
        self.adjacency_mat = adjacency_mat
        self.lambda_ = lambda_  # Population size
        self.mu = mu  # Offspring size (must be the double of lambda)
        self.alpha = alpha  # Mutation probability
        self.max_iterations = max_iterations  # Maximum number of iterations
        self.MAX_DIFFERENT_BEST_SOLUTIONS = max_iterations / 5

    def tourIsValid(self, tour: np.ndarray):
        # Check if the tour is valid
        if len(np.unique(tour[:self.adjacency_mat.shape[0]])) != self.adjacency_mat.shape[0]:
            return False
        else:
            return True

    def objf(self, cities: np.ndarray) -> int:
        # Apply the objective function to each row of the cities array
        cities = cities.astype(int)
        sum = 0
        for ii in range(self.adjacency_mat.shape[0] - 1):
            # Sum the distances between the cities
            sum += self.adjacency_mat[cities[ii], cities[ii + 1]]

        # Add the distance between the last and first city
        sum += self.adjacency_mat[cities[self.adjacency_mat.shape[0] - 1], cities[0]]

        return sum
    
    """ Initialize the population with random individuals. """
    # DFS parti da 0 e guardo quali sono i nodi che non hanno infinito e ne prendo uno a caso, per ogni nodo
    def initialize(self) -> None:
        # Create a matrix of random individuals
        population = np.zeros((self.lambda_, self.adjacency_mat.shape[0] + 1)) # +1 for the alpha value
        for i in range(self.lambda_):
            # Create alpha with gaussian distribution
            alpha = np.random.normal(self.alpha, 0.15)
            # Create a random individual and concatenate alpha
            rIndividual = np.concatenate(([0], np.random.permutation(np.arange(1, self.adjacency_mat.shape[0])), [alpha]))
            # Evaluate the individual with the objective function
            obj = self.objf(rIndividual)
            # Check if the individual is valid
            while obj == np.inf:
                # Create a random individual
                rIndividual = np.concatenate(([0], np.random.permutation(np.arange(1, self.adjacency_mat.shape[0])), [alpha]))
                # Evaluate the individual with the objective function
                obj = self.objf(rIndividual)

            population[i, :] = rIndividual

        return population

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
    
    
    

    def two_opt(self, population: np.ndarray, k):

        if k < 1 or k > self.adjacency_mat.shape[0] - 1:
            raise ValueError("k must be between 2 and n-1")

        for i in range(population.shape[0]):

            # For each individual in the population
            best_tour = population[i, :]
            best_obj = self.objf(best_tour)


            for j in range(k):
                # Select two random indices
                ri = sorted([random.randrange(1, self.adjacency_mat.shape[0] - 1), random.randrange(1, self.adjacency_mat.shape[0] - 1)])
                #print("ri: ", ri)
                tour = population[i, :].copy()

                if not self.tourIsValid(tour):
                    raise ValueError("Duplicate in tour durin two opt")

                tour = tour.astype(int)
                # Connect ri[0] with ri[1] and ri[0]+1 with ri[1]+1
                tour[:ri[0]] = population[i, :ri[0]]
                tour[ri[0]:ri[1]] = population[i, ri[0]:ri[1]]
                tour[ri[1]:] = population[i, ri[1]:]

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

    """ Perform k-tournament selection WITHOUT REPLACEMNT to select pairs of parents. """
    def selection(self, population: np.ndarray, k: int = 3):

        # Create a matrix of selected parents
        selected = np.zeros((self.mu * 2, self.adjacency_mat.shape[0] + 1)) # +1 for the alpha value

        # Selecting the parents
        for ii in range(self.mu * 2):
            # Select k random individuals from the population
            ri = random.sample(range(population.shape[0]), k)

            # Select the best individual from the k random individuals
            min = np.argmin(np.apply_along_axis(self.objf, 1, population[ri, :]))

            # Add the selected individual to the matrix of selected parents
            selected[ii, :] = population[ri[min], :]
        return selected
    
    def pmx (self, parent1, parent2):
        # Create a child
        child1 = np.ones(shape = self.adjacency_mat.shape[0])
        child1 = child1 * -1
        # select random start and end indices for parent1's subsection
        start, end = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])
        # copy parent1's subsection into child
        child1[start:end] = parent1[start:end]
        # fill the remaining positions in order
        child1[child1 == -1] = [i for i in parent2[:-1] if i not in child1]

        child2 = np.ones(shape = self.adjacency_mat.shape[0])
        child2 = child2 * -1
        # select random start and end indices for parent1's subsection
        start, end = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])
        # copy parent2's subsection into child
        child2[start:end] = parent2[start:end]
        # fill the remaining positions in order
        child2[child2 == -1] = [i for i in parent1[:-1] if i not in child2]

        # Add the alpha value to the child (average of the parents' alpha values)
        new_alpha = (parent1[-1] + parent2[-1]) / 2

        # Concatenate the child to the alpha value
        child1 = np.concatenate((child1, [new_alpha]))
        child2 = np.concatenate((child2, [new_alpha]))

        return child1, child2

    """ Perform crossover"""   
    def crossover(self, selected: np.ndarray):
        # Create a matrix of offspring
        offspring = np.zeros((self.mu, self.adjacency_mat.shape[0] + 1)) # +1 for the alpha value

        for ii in range(self.lambda_):
            # Select two random parents
            ri = sorted([random.randrange(0, self.lambda_), random.randrange(0, self.lambda_)])

            # Perform crossover
            offspring[ii, :], offspring[ii + self.lambda_, :] = self.pmx(selected[ri[0], :], selected[ri[1], :])

            # Check if the children are valid
            if not self.tourIsValid(offspring[ii, :]) or not self.tourIsValid(offspring[ii + self.lambda_, :]):
                raise ValueError("Duplicate in offspring durin crossover")

        return offspring

    def swap_mutation(self, offspring,ii):
        # Two positions (genes) in the chromosome are selected at
        # random and their allele values swapped

        # Select two random indices
        ri = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])

        # Swap the cities
        offspring[ii, ri[0]], offspring[ii, ri[1]] = (
            offspring[ii, ri[1]],
            offspring[ii, ri[0]],
        )

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :-1])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin swap mutation")
        

    def insert_mutation(self,offspring,ii):
        # Two alleles are selected at random and the second moved
        # next to the first, shuffling along the others to make room

        # Select two random indices sorted, they must be different
        ri = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])

        if ri[0] == ri[1]:
            if ri[0] == 0:
                ri[1] += 1
            elif ri[0] == self.adjacency_mat.shape[0]-1:
                ri[0] -= 1
            else:
                ri[0] -= 1

        # Shift the cities from the first index to the second index
        app = offspring[ii, ri[1]-1]
        offspring[ii, ri[0] + 1 : ri[1]] = offspring[ii, ri[0] : ri[1] - 1]

        # Insert the city at the first index at the second index
        offspring[ii, ri[0]] = app

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :-1])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin insert mutation")


    def scramble_mutation(self,offspring,ii):
        # a randomly chosen subset of values 
        # are chosen and their order randomly shuffled

        # Select two random indices sorted
        ri = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])

        # Shuffle the cities
        np.random.shuffle(offspring[ii, ri[0]:ri[1]])

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :-1])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin scramble mutation")


    def inversion_mutation(self,offspring,ii):
        # Randomly select two positions in the chromosome and
        # reverse the order in which the values appear between those positions

        # Select two random indices sorted
        ri = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])

        # Invert the cities
        offspring[ii, ri[0]:ri[1]] = offspring[ii, ri[0]:ri[1]][::-1]

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :-1])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin inversion mutation")

    def swap_longest_links_mutation(self, individual: np.ndarray):
        # Select the longest links in the individual and swap them

        individual = individual.astype(int)
        # Calculate the distance between each city
        distances = np.zeros(self.adjacency_mat.shape[0])
        for ii in range(self.adjacency_mat.shape[0] - 1):
            distances[ii] = self.adjacency_mat[individual[ii], individual[ii + 1]]

        # Select the longest links
        longest_links = np.argsort(distances)[-2:]

        # Swap the longest links
        individual[longest_links[0]], individual[longest_links[1]] = (
            individual[longest_links[1]],
            individual[longest_links[0]],
        )

        if len(np.unique(individual[:-1])) != self.adjacency_mat.shape[0]:
            print("Duplicate in offspring durin swap longest links mutation")

        return individual

    def adapt_alpha(self, individual: np.ndarray):
        # Calculate success rate based on the objective function value
        success_rate = (1.0 / (np.sqrt(self.objf(individual)) + 1)) * self.adjacency_mat.shape[0]


        # Update alpha using a simple evolutionary strategy
        if success_rate > 0.85:
            individual[self.adjacency_mat.shape[0]] *= 0.9
        elif success_rate < 0.4:
            individual[self.adjacency_mat.shape[0]] *= 1.1

        # Ensure alpha stays within a reasonable range (0 to 1)
        individual[self.adjacency_mat.shape[0]] = max(0.01, min(0.99, individual[self.adjacency_mat.shape[0]]))


    """ Perform mutation on the offspring."""

    def mutation(self, offspring: np.ndarray):

        # Apply the mutation to each row of the offspring array
        for ii, _ in enumerate(offspring):
            # Update alpha based on the success of the individual
            self.adapt_alpha(offspring[ii, :])

            # Apply the mutation with probability alpha of the individual
            if np.random.rand() < offspring[ii, self.adjacency_mat.shape[0]]:

                # Add a noise to the alpha value
                offspring[ii, -1] = np.random.normal(self.alpha, 0.02)
                            
                # Randomly select a mutation operator with different probabilities
                mutation_operator = np.random.choice(np.arange(0,5), 1, p=[0.55, 0.04, 0.03, 0.03, 0.35])

                if mutation_operator == 0:
                    self.inversion_mutation(offspring,ii)
                elif mutation_operator == 1:
                    self.swap_longest_links_mutation(offspring[ii, :])
                elif mutation_operator == 2:
                    self.scramble_mutation(offspring,ii)
                elif mutation_operator == 3:
                    self.insert_mutation(offspring,ii)
                else:
                    self.swap_mutation(offspring,ii)

        return offspring

    """ Eliminate the unfit candidate solutions. """

    def elimination(self, joinedPopulation: np.ndarray):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = np.apply_along_axis(self.objf, 1, joinedPopulation)
        # Sort the individuals based on their objective function value
        perm = np.argsort(fvals)
        # Select the best lambda individuals
        survivors = joinedPopulation[perm[0 : self.lambda_], :]

        return survivors

        
    def elimination_with_crowding(self, joinedPopulation: np.ndarray):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = np.apply_along_axis(self.objf, 1, joinedPopulation)

        # Calculate the crowding distance for each individual
        crowding_distances = self.calculate_crowding_distances(joinedPopulation, fvals)

        # Sort the individuals based on the objective function value and crowding distance
        sorted_indices = np.lexsort((crowding_distances, fvals))
        sorted_population = joinedPopulation[sorted_indices, :]

        # Select the best lambda individuals with crowding selection
        survivors = sorted_population[:self.lambda_, :]

        return survivors

    def calculate_crowding_distances(self, population: np.ndarray, fvals: np.ndarray):
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


# Main 
if __name__ == "__main__":
    # Calculate the time
    start_time = time.time()
    r0123456().optimize("tour200.csv")
    print("--- %s seconds ---" % (time.time() - start_time))


# Benchmark results to beat:
# tour50: simple greedy heuristic 27723
# tour100: simple greedy heuristic 90851
# tour200: simple greedy heuristic 39745 (BEST 52k) [std,two_opt,one_opt] [alpha=adapt, sigma_share=0.1, k=3, lambda=200, mu=200]
# tour500: simple greedy heuristic 157034
# tour750: simple greedy heuristic 197541
# tour1000: simple greedy heuristic 195848
