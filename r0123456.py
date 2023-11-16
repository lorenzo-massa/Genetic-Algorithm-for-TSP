import Reporter
import numpy as np
import random
import matplotlib.pyplot as plt


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

        TSP = TravelsalesmanProblem(distanceMatrix)
        population = TSP.initialize()
        iteration = 0

        # Store the progress
        meanObjective = 0.0
        bestObjective = 0.0
        bestSolution = np.array([1, 2, 3, 4, 5])

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
            selected = TSP.selection(population)
            offspring = TSP.crossover(selected)
            joinedPopulation = np.vstack(
                (TSP.mutation(offspring), TSP.mutation(population))
            )
            population = TSP.elimination(joinedPopulation)

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

            # Print progress
            print(
                "Iteration: %d, Mean: %f, Best: %f"
                % (iteration, meanObjective, bestObjective)
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
            if timeLeft < 0:
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
        lambda_: int = 500,
        mu: int = 500,
        k: int = 3,
        alpha: float = 0.2,
        max_iterations= 100,
    ) -> None:
        self.adjacency_mat = adjacency_mat
        self.lambda_ = lambda_  # Population size
        self.mu = mu  # Offspring size
        self.k = k  # Tournament selection
        self.alpha = alpha  # Mutation probability
        self.max_iterations = max_iterations  # Maximum number of iterations
        self.MAX_DIFFERENT_BEST_SOLUTIONS = 20

    def objf(self, cities: np.ndarray) -> int:
        # Apply the objective function to each row of the cities array
        cities = cities.astype(int)
        sum = 0
        for ii in range(cities.shape[0] - 1):
            # Sum the distances between the cities
            sum += self.adjacency_mat[cities[ii], cities[ii + 1]]

        # Add the distance between the last and first city
        sum += self.adjacency_mat[cities[-1], cities[0]]

        return sum

    def initialize(self) -> None:
        # Create a matrix of random individuals
        population = np.zeros((self.lambda_, self.adjacency_mat.shape[0]))
        for i in range(self.lambda_):
            # Create a random individual
            rIndividual = np.random.permutation(np.arange(self.adjacency_mat.shape[0]))
            # Evaluate the individual with the objective function
            obj = self.objf(rIndividual)
            # Check if the individual is valid
            while obj == np.inf:
                # Create a random individual
                rIndividual = np.random.permutation(np.arange(self.adjacency_mat.shape[0]))
                # Evaluate the individual with the objective function
                obj = self.objf(rIndividual)

            population[i, :] = rIndividual

        return population

                
        

    """ Perform k-tournament selection WITH REPLACEMENT to select pairs of parents. """

    def selection_with_replacement(self, population: np.ndarray):
        # Create a matrix of selected parents
        selected = np.zeros((self.mu * 2, self.adjacency_mat.shape[0]))
        # Selecting the parents
        for ii in range(self.mu * 2):
            # Select k random individuals from the population
            ri = random.choices(range(np.size(population, 1)), k=self.k)

            # Select the best individual from the k random individuals
            min = np.argmin(np.apply_along_axis(self.objf, 1, population[ri, :]))

            # Add the selected individual to the matrix of selected parents
            selected[ii, :] = population[ri[min], :]
        return selected

    """ Perform k-tournament selection WITHOUT REPLACEMNT to select pairs of parents. """
    def selection(self, population: np.ndarray):
        # Create a matrix of selected parents
        selected = np.zeros((self.mu * 2, self.adjacency_mat.shape[0]))
        # Selecting the parents
        for ii in range(self.mu * 2):
            # Select k random individuals from the population
            ri = random.sample(range(np.size(population, 1)), k=self.k)

            # Select the best individual from the k random individuals
            min = np.argmin(np.apply_along_axis(self.objf, 1, population[ri, :]))

            # Add the selected individual to the matrix of selected parents
            selected[ii, :] = population[ri[min], :]
        return selected


    """ Perform crossover"""   
    def crossover(self, selected: np.ndarray):
        # Create a matrix of offspring
        offspring = np.zeros((self.mu, self.adjacency_mat.shape[0]))

        for ii, _ in enumerate(offspring):

            # Create a child
            child = np.ones(shape = self.adjacency_mat.shape[0])
            child = child * -1
            # select random start and end indices for parent1's subsection
            start, end = sorted([random.randrange(self.adjacency_mat.shape[0]), random.randrange(self.adjacency_mat.shape[0])])
            # copy parent1's subsection into child
            child[start:end] = selected[2 * ii, start:end]
            # fill the remaining positions in order
            child[child == -1] = [i for i in selected[2 * ii + 1, :] if i not in child]
            
            offspring[ii, :] = child

            """ Debugging """
            # print("----------------------------------------")
            # print("ii: ", ii)
            # print("first parent: ", selected[2*ii, :])
            # print("second parent: ", selected[2*ii+1, :])
            # print("first half: ", first_half)
            # print("second half: ", second_half)
            # print("child: ", offspring[ii,:])

        return offspring

    def swap_mutation(self, offspring,ii):
        # Two positions (genes) in the chromosome are selected at
        # random and their allele values swapped

        # Select two random indices
        ri = sorted([random.randrange(self.adjacency_mat.shape[0]), random.randrange(self.adjacency_mat.shape[0])])

        # Swap the cities
        offspring[ii, ri[0]], offspring[ii, ri[1]] = (
            offspring[ii, ri[1]],
            offspring[ii, ri[0]],
        )

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin swap mutation")
        

    def insert_mutation(self,offspring,ii):
        # Two alleles are selected at random and the second moved
        # next to the first, shuffling along the others to make room

        # Select two random indices sorted, they must be different
        ri = sorted([random.randrange(self.adjacency_mat.shape[0]), random.randrange(self.adjacency_mat.shape[0])])

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
        if len(np.unique(offspring[ii, :])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin insert mutation")


    def scramble_mutation(self,offspring,ii):
        # a randomly chosen subset of values 
        # are chosen and their order randomly shuffled

        # Select two random indices sorted
        ri = sorted([random.randrange(self.adjacency_mat.shape[0]), random.randrange(self.adjacency_mat.shape[0])])

        # Shuffle the cities
        np.random.shuffle(offspring[ii, ri[0]:ri[1]])

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin scramble mutation")


    def inversion_mutation(self,offspring,ii):
        # Randomly select two positions in the chromosome and
        # reverse the order in which the values appear between those positions

        # Select two random indices sorted
        ri = sorted([random.randrange(self.adjacency_mat.shape[0]), random.randrange(self.adjacency_mat.shape[0])])

        # Invert the cities
        offspring[ii, ri[0]:ri[1]] = offspring[ii, ri[0]:ri[1]][::-1]

        # Check if the offspring has duplicates
        if len(np.unique(offspring[ii, :])) != self.adjacency_mat.shape[0]:
            print("offspring: ", offspring[ii, :]," after")
            print("Duplicate in offspring durin inversion mutation")




    """ Perform mutation on the offspring."""

    def mutation(self, offspring: np.ndarray):
        """
        Lots of optimization can be done here!!!!
        """
        # Apply the mutation to each row of the offspring array
        for ii, _ in enumerate(offspring):
            # Apply the mutation with probability alpha
            if np.random.rand() < self.alpha:
                
                # Randomly select a mutation operator with different probabilities
                mutation_operator = np.random.choice(np.arange(0,4), 1, p=[0.4, 0.2, 0.2, 0.2])

                if mutation_operator == 0:
                    self.inversion_mutation(offspring,ii)
                elif mutation_operator == 1:
                    self.insert_mutation(offspring,ii)
                elif mutation_operator == 2:
                    self.scramble_mutation(offspring,ii)
                else:
                    self.swap_mutation(offspring,ii)

        return offspring

    """ Perform mutation on the offspring."""

    # Mutation applied to each individual
    def mutation_for_each(self, offspring: np.ndarray):
        # Apply the mutation to each row of the offspring array
        for ii, _ in enumerate(offspring):
            for jj, _ in enumerate(offspring[ii, :]):
                # Apply the mutation with probability alpha
                if np.random.rand() < self.alpha*0.1:
                    # Select a random index
                    ri = np.random.choice(
                        range(self.adjacency_mat.shape[0]), 1, replace=False
                    )
                    # Swap the actual city with the random city
                    offspring[ii, jj], offspring[ii, ri[0]] = (
                        offspring[ii, ri[0]],
                        offspring[ii, jj],
                    )
                elif np.random.rand() < 0.03:
                    """Heu ... should we actually do this?"""
                    # Randomly create a new individual
                    offspring[ii, :] = np.random.permutation(
                        range(self.adjacency_mat.shape[0])
                    )

        return offspring

    def mutation_sublist(self, offspring: np.ndarray):
        # Apply the mutation to each row of the offspring array
        for ii, _ in enumerate(offspring):
            # Apply the mutation with probability alpha
            if np.random.rand() < self.alpha:
                # Get two random indices
                ri = np.random.choice(range(self.adjacency_mat.shape[0]), 2)
                # If the first index is smaller than the second index, do normal sublist permutation
                if ri[0] < ri[1]:
                    # Slice from start_index to end_index (exclusive)
                    subarray = offspring[ii, ri[0] : ri[1]]
                    np.random.shuffle(subarray)
                    offspring[ii, ri[0] : ri[1]] = subarray
                # If the second index is smaller (or equal) than the first index, the list is taken from the
                # first index until the end of the list and then from the start of the list until the
                # second index
                else:
                    # Slice from start_index to end_index (exclusive)
                    subarray = np.concatenate(
                        (offspring[ii, ri[0] :], offspring[ii, : ri[1]])
                    )
                    np.random.shuffle(subarray)
                    offspring[ii, :] = np.concatenate(
                        (subarray, offspring[ii, ri[1] : ri[0]])
                    )
        return offspring

    """ Eliminate the unfit candidate solutions. """

    def elimination(self, joinedPopulation: np.ndarray):
        # Apply the objective function to each row of the joinedPopulation array
        fvals = np.apply_along_axis(self.objf, 1, joinedPopulation)
        # Sort the individuals based on their objective function value
        perm = np.argsort(fvals)
        # Select the best lambda individuals
        survivors = joinedPopulation[perm[0 : self.lambda_], :]

        """ Debugging """
        # print("----------------------------------------")
        # print("joinedPopulation: ", joinedPopulation.shape)
        # print("fvals: ", fvals.shape)
        # print("perm: ", perm.shape)
        # print("survivors: ", survivors.shape)

        return survivors


# Main 
if __name__ == "__main__":
    r0123456().optimize("cost_matrix.csv")
