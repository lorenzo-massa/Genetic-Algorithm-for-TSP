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
                                    alpha=0.3,
                                    max_iterations=200)
        

        # Initialize the population
        start_time = time.time()
        population = TSP.initialize()
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
            selected = TSP.selection(population,3)

            # Crossover
            offspring = TSP.crossover(selected)

            joinedPopulation = np.vstack(
                (TSP.mutation(offspring), TSP.mutation(population))
            )

            # Elimination
            if iteration < 20:
                population = TSP.elimination_with_crowding(joinedPopulation)
            elif iteration >= 20 and iteration < 40:
                joinedPopulation = TSP.one_opt(joinedPopulation, 3)
                population = TSP.elimination_with_crowding(joinedPopulation)
            elif iteration >= 40 and iteration < 60:
                joinedPopulation = TSP.one_opt(joinedPopulation, 6)
                population = TSP.elimination_with_crowding(joinedPopulation)
            else:
                joinedPopulation = TSP.one_opt(joinedPopulation, 9)
                population = TSP.elimination_with_crowding(joinedPopulation)


            # Show progress
            fvals = np.apply_along_axis(TSP.objf, 1, population)
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
        self.bestObj = np.inf

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
    def initialize(self) -> None:
        # Create a matrix of random individuals
        population = np.zeros((self.lambda_, self.adjacency_mat.shape[0] + 1)) # +1 for the alpha value

        # Create alpha with gaussian distribution
        alpha = np.random.normal(self.alpha, 0.05)
        # Create the first individual with greedy heuristic
        population[0, :] = np.concatenate((self.generate_individual_greedy(), [alpha]))

        for i in range(1, self.lambda_):
            
            # Create an individual
            if i < self.lambda_/20:
                new_individual = self.generate_individual_nearest_neighbor()
            else:
                new_individual = self.generate_individual_random()

            
            # Evaluate the individual with the objective function
            obj = self.objf(new_individual)
            # Check if the individual is valid for at most 100 times
            max_tries = self.adjacency_mat.shape[0]/5
            while obj == np.inf and max_tries > 0:
                # Create a random individual
                new_individual = self.generate_individual_random()
                # Evaluate the individual with the objective function
                obj = self.objf(new_individual)
                max_tries -= 1


            # Create alpha with gaussian distribution
            alpha = np.random.normal(self.alpha, 0.05)
            # Concatenate the alpha value to the individual
            new_individual = np.concatenate((new_individual, [alpha]))

            population[i, :] = new_individual

        return population
    
    def generate_individual_greedy(self):
        # Create an individual choosing always the nearest city
        individual = np.zeros(self.adjacency_mat.shape[0]).astype(int)
        individual[0] = 0

        not_visited = np.arange(1, self.adjacency_mat.shape[0])

        for ii in range(1, self.adjacency_mat.shape[0]):
            # Select the nearest city
            nearest_city = np.argmin(self.adjacency_mat[individual[ii - 1].astype(int), not_visited])
            # Add the nearest city to the individual
            individual[ii] = not_visited[nearest_city]
            # Remove the nearest city from the not visited list
            not_visited = np.delete(not_visited, nearest_city)

        return individual
    
    def generate_individual_nearest_neighbor(self):
        # Create an individual choosing always the nearest city , second city is random
        individual = np.zeros(self.adjacency_mat.shape[0]).astype(int)
        individual[0] = 0
        individual[1] = np.random.randint(1, self.adjacency_mat.shape[0]) # Second city is random

        not_visited = np.arange(1, self.adjacency_mat.shape[0])
        not_visited = np.delete(not_visited, np.where(not_visited == individual[1]))

        for ii in range(2, self.adjacency_mat.shape[0]):
            # Select the nearest city
            nearest_city = np.argmin(self.adjacency_mat[individual[ii - 1].astype(int), not_visited])
            # Add the nearest city to the individual
            individual[ii] = not_visited[nearest_city]
            # Remove the nearest city from the not visited list
            not_visited = np.delete(not_visited, nearest_city)

        return individual
    

    def generate_individual_random(self):
        return np.concatenate(([0], np.random.permutation(np.arange(1, self.adjacency_mat.shape[0])).astype(int)))



    def one_opt(self, population: np.ndarray, k):

        modified = 0

        if k < 1 or k > population.shape[0] - 1:
            raise ValueError("k must be between 2 and n-1")

        for i in range(population.shape[0]):

            # For each individual in the population
            best_tour = population[i, :]
            best_obj = self.objf(best_tour)

            # Select k random indices
            k_list = sorted(random.sample(range(1, self.adjacency_mat.shape[0]), k))

            for ri in k_list:
                # Swap the ri-th and ri+1-th cities
                tour = population[i, :].copy()
                tour[ri], tour[ri+1] = tour[ri+1], tour[ri]

                # Evaluate the new tour
                new_obj = self.objf(tour)

                # Check if the new tour is better
                if new_obj < best_obj:
                    best_tour = tour
                    best_obj = new_obj

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
    
    def one_opt_mining(self, population: np.ndarray):

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
        selected = np.zeros((self.mu, self.adjacency_mat.shape[0] + 1))  # +1 for the alpha value
        selected[:,:-1] = selected[:,:-1].astype(int)

        # Selecting the parents
        for ii in range(self.mu):
            # Select k random individuals from the population
            ri = random.sample(range(population.shape[0]), k)

            # Select the best individual from the k random individuals
            best = np.argmin(np.apply_along_axis(self.objf, 1, population[ri, :]))

            # Add the selected individual to the matrix of selected parents
            selected[ii, :] = population[ri[best], :]
        return selected
    
    def pmx(self, parent1, parent2):
        # Create a child
        child1 = np.ones(shape = self.adjacency_mat.shape[0]).astype(int)
        child1 = child1 * -1
        # select random start and end indices for parent1's subsection
        start, end = sorted([random.randrange(1, self.adjacency_mat.shape[0]), random.randrange(1, self.adjacency_mat.shape[0])])
        # copy parent1's subsection into child
        child1[start:end] = parent1[start:end]
        # fill the remaining positions in order
        child1[child1 == -1] = [i for i in parent2[:-1] if i not in child1]

        child2 = np.ones(shape = self.adjacency_mat.shape[0]).astype(int)
        child2 = child2 * -1
        # use the same random start and end indices for parent2's subsection
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
        offspring[:,:-1] = offspring[:,:-1].astype(int)

        for ii in range(self.lambda_):
            # Select two random parents
            ri = sorted(random.sample(range(1, self.lambda_), k = 2))

            # Perform crossover
            offspring[ii, :], offspring[ii + self.lambda_, :] = self.pmx(selected[ri[0], :], selected[ri[1], :])

        return offspring

    def swap_mutation(self, tour):
        # Select two random indices
        ri = sorted(random.sample(range(1, self.adjacency_mat.shape[0]), k = 2))

        # Swap the cities
        tour[ri[0]], tour[ri[1]] = tour[ri[1]], tour[ri[0]]

        return tour
        

    def insert_mutation(self,tour):
        # Select two random indices sorted, they must be different
        ri = sorted(random.sample(range(1, self.adjacency_mat.shape[0]), k = 2))

        np.delete(tour, ri[0])
        np.insert(tour, ri[1], tour[ri[0]])

        return tour


    def scramble_mutation(self,tour):
        # Select two random indices sorted
        ri = sorted(random.sample(range(1, self.adjacency_mat.shape[0]), k = 2))

        # Shuffle the cities between the two indices
        np.random.shuffle(tour[ri[0]:ri[1]])

        return tour


    def inversion_mutation(self,tour):
        # Select two random indices sorted
        ri = sorted(random.sample(range(1, self.adjacency_mat.shape[0]), k = 2))

        # Invert the cities
        tour[ri[0]:ri[1]] = tour[ri[0]:ri[1]][::-1]

        return tour

    def swap_longest_links_mutation(self, individual: np.ndarray):
        # Select the two longest links in the individual and swap them

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

        return individual

    def adapt_alpha(self, individual: np.ndarray):
        # Calculate success rate based on the objective function value
        success_rate = self.objf(individual) / self.bestObj


        # Update alpha using a simple evolutionary strategy
        if success_rate > 0.9:
            individual[self.adjacency_mat.shape[0]] *= 0.9
        elif success_rate < 0.5:
            individual[self.adjacency_mat.shape[0]] *= 1.1

        # Ensure alpha stays within a reasonable range (0 to 1)
        individual[self.adjacency_mat.shape[0]] = max(0.27, min(0.99, individual[self.adjacency_mat.shape[0]]))


    """ Perform mutation on the offspring."""

    def mutation(self, offspring: np.ndarray):

        # Apply the mutation to each row of the offspring array
        for ii, _ in enumerate(offspring):
            # Update alpha based on the success of the individual
            #self.adapt_alpha(offspring[ii, :])

            # Apply the mutation with probability alpha of the individual
            if np.random.rand() < offspring[ii, self.adjacency_mat.shape[0]]:

                # Add a noise to the alpha value
                offspring[ii, -1] = np.random.normal(self.alpha, 0.02)
                            
                # Randomly select a mutation operator with different probabilities
                mutation_operator = random.choices([self.inversion_mutation,self.swap_mutation,
                    self.scramble_mutation,self.insert_mutation], weights=[9, 4, 1, 1], k=1)[0]
                
                
                offspring[ii, :] = mutation_operator(offspring[ii, :])

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


# Main 
if __name__ == "__main__":
    # Calculate the time
    start_time = time.time()
    r0123456().optimize("tour500.csv")
    print("--- %s seconds ---" % (time.time() - start_time))


# Benchmark results to beat:
# tour50: simple greedy heuristic 27723     (TARGET 24k)    (BEST 26k)
# tour100: simple greedy heuristic 90851    (TARGET 81k)    (BEST 81k)
# tour200: simple greedy heuristic 39745    (TARGET 35k)    (BEST 40k)
# tour500: simple greedy heuristic 157034   (TARGET 141k)   (BEST 155k)
# tour750: simple greedy heuristic 197541
# tour1000: simple greedy heuristic 195848
