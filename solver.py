# --- Imports ------------------------------------------------------------------

# Standard library
import time

# Third party libraries
import cvxpy as cp
import gurobipy as gp
import numpy as np


# --- Solver -------------------------------------------------------------------

class Solver():
    def __init__(self, algorithm, loss=None):
        """
        Initialize a solver with a solve algorithm and a loss function.

        Args:
            algorithm (function): The solve algorithm           
            loss (object, optional): The loss function class object
        """
        self.algorithm = algorithm
        self.loss = loss

    def solve(self, dataset, solveArray="dataArray", **parameters):
        """
        Solve for optimal with algorithm and loss function for specified dataset

        Args:
            dataset (object): The dataset class object  
            **parameters: Additional parameters of the algorithm function          
        """
        if self.loss is None:
            return self.algorithm(dataset = dataset, 
                                  environment = self.environment, 
                                  **parameters)
        else:
            return self.algorithm(dataset = dataset, 
                                  lossFunction = self.loss,
                                  solveArray = solveArray, 
                                  **parameters)
    
    def createEnvironment(self, outputFlag=0):
        """
        Create and set up a gp environment for integer optimization solvers.

        Arg: outputFlag (int): Flag.
        """
        self.environment = gp.Env(empty=True)
        self.environment.setParam("OutputFlag", outputFlag)
        self.environment.start()


# --- Utility ------------------------------------------------------------------

def randomSample(datasetSize, subsetSize, seed=None):
    """
    Randomly sample from dataset by generating random indices to create subset

    Args:
        datasetSize (tuple): The dataset dimensions sizes
        subsetSize (int): The number of points to sample for the subset
        seed (int, rng, optional): The random seed or Numpy rng for random 
            generation and reproducibility
    
    Returns:
        z (array): indicator vector of included items in the subset
        indices (array): indices in the dataset of included items in the subset
    """  
    rng = np.random.default_rng(seed)
    indices = rng.choice(datasetSize[0], size=subsetSize, replace=False)
    z = np.zeros(datasetSize[0])
    z[indices] = 1
    return z, indices

def optimize(objective, constraints, environment, solver=cp.SCIP, 
             verbose=False):
    """
    Sets up a cvxpy problem with given objective and constraints and solves it 
    using the specified solver.

    Args:
        objective: cvxpy Objective object defining the optimization objective.
        constraints: List of cvxpy Constraint objects defining the optimization 
            constraints.
        environment: Optional. Environment or settings required by the solver, 
            particularly when using external solvers like Gurobi.
        solver: Optional. Solver to be used for solving the optimization 
            problem. Defaults to cp.GUROBI. Other possible solvers include 
            cp.ECOS, cp.OSQP, etc.
        verbose: Optional. Boolean flag indicating whether to print solver 
            output messages during optimization. Defaults to False.

    Returns: problem: The cvxpy Problem object after solving, which contains 
        solution information and other attributes.
    """
    problem = cp.Problem(objective, constraints)
    if solver == cp.GUROBI:
        problem.solve(solver=solver, verbose=verbose, env=environment)
    else:
        problem.solve(solver=solver, verbose=verbose)
    return problem


# --- Algorithms ---------------------------------------------------------------

def bestOfRandom(dataset, lossFunction, subsetSize, solveArray, minLoss=0, 
               maxIterations=None, seed=None, verbose=False, selectBy="row"):
    time0 = time.time()

    if maxIterations is None:
        maxIterations = dataset.size[0]

    z = randomSample(dataset.size, subsetSize, seed)[0]
    minLoss = lossFunction.calculate(dataset, z)

    for i in range(maxIterations):
        curZ = randomSample(dataset.size, subsetSize, seed)[0]
        curLoss = lossFunction.calculate(dataset, curZ)
        if curLoss < minLoss:
            z = curZ
            minLoss = curLoss

    time1 = time.time()
    timeTotal = time1 - time0

    return z, timeTotal, minLoss


def averageOfRandom(dataset, lossFunction, subsetSize, solveArray, minLoss=0, 
               maxIterations=None, seed=None, verbose=False, selectBy="row"):
    time0 = time.time()

    if maxIterations is None:
        maxIterations = dataset.size[0]

    z = randomSample(dataset.size, subsetSize, seed)[0]
    losses = [lossFunction.calculate(dataset, z)]

    for i in range(maxIterations):
        curZ = randomSample(dataset.size,subsetSize, seed)[0]
        losses.append(lossFunction.calculate(dataset, curZ))

    avgLoss = np.mean(losses)

    time1 = time.time()
    timeTotal = time1 - time0

    return z, timeTotal, avgLoss


def worstOfRandom(dataset, lossFunction, subsetSize, solveArray, minLoss=0, 
               maxIterations=None, seed=None, verbose=False, selectBy="row"):
    """
    maximize representativeness of a subset of size s of dataset of size n by m
    according to metric function f using the p-norm
    """
    time0 = time.time()

    if maxIterations is None:
        maxIterations = dataset.size[0]

    z = randomSample(dataset.size, subsetSize, seed)[0]
    maxLoss = lossFunction.calculate(dataset, z)

    for i in range(maxIterations):
        curZ = randomSample(dataset.size, subsetSize, seed)[0]
        curLoss = lossFunction.calculate(dataset, curZ)
        if curLoss > maxLoss:
            z = curZ
            maxLoss = curLoss

    time1 = time.time()
    timeTotal = time1 - time0

    return z, timeTotal, maxLoss


def greedySwap(dataset, lossFunction, subsetSize, solveArray, minLoss=0, 
               maxIterations=None, seed=None, verbose=False, selectBy="row"):
    """
    A greedy algorithm with a greedy swap heuristic for subset selection.

    Args:
        dataset (object): The Dataset class object
        lossFunction (object): The loss function class object
        subsetSize (int): The desired subset size
        solveArray (string): The desired array to use during solving
        minLoss (float): The minimum loss value to stop iterations
        maxIterations (int, optional): Maximum number of iterations
        seed (int, rng, optional): The random seed or NumPy rng for random 
            generation and reproducibility
        verbose (bool, optional): Toggle for verbose logging

    Returns:
        z (array): Indicator vector of included items in the subset
        timeTotal (float): Total execution time
        loss (float): The loss value of the final subset
    """
    if verbose:
        print(f"Solving for a subset of size {subsetSize} with "
              f"{lossFunction.objective.__name__} objective on {solveArray}.")
    time0 = time.time() # get start time
    iterations = 0

    # select random starting subset
    z, indices = randomSample(dataset.size, subsetSize, seed)
    loss = lossFunction.calculate(dataset, z)

    if maxIterations is None:
        maxIterations = dataset.size[0]

    for i in range(maxIterations):
        if verbose:
            print(f"Iteration {i}: Loss {loss}")
        if i not in indices:
            zSwapBest = np.copy(z)
            lossSwapBest = loss
            indicesSwapBest = np.copy(indices)

            for loc, j in enumerate(indices):
                iterations += 1
                zSwap = np.copy(z)
                zSwap[i] = 1 # add the new datapoint
                zSwap[j] = 0 # remove the old datapoint
                indicesSwap = np.copy(indices)
                indicesSwap[loc] = i
                lossSwap = lossFunction.calculate(dataset, zSwap)

                if lossSwap < lossSwapBest:
                    zSwapBest = np.copy(zSwap)
                    lossSwapBest = lossSwap
                    indicesSwapBest = indicesSwap

            z = np.copy(zSwapBest)
            loss = lossSwapBest
            indices = indicesSwapBest

            if loss == minLoss:
                break

    time1 = time.time()       # get end time
    timeTotal = time1 - time0 # calculate total time

    return z, timeTotal, loss # return indices, total time and loss


def optimizeCoverage(dataset, environment, subsetSize, verbose=False):
    """
    Optimize subset selection for coverage while minimizing L1 norm.

    Args:
        environment: The environment or solver settings for optimization.
        datasetOnehot (numpy.ndarray): The one-hot encoded dataset.
        subsetSize (int): The desired size of the subset.

    Returns:
        z (numpy.ndarray): Binary array indicating the selected subset.
        timeTotal (float): Total time taken for optimization.
        problem.value (float): The value of the optimization problem.
    """ 
    time_0 = time.time()

    datasetLength, oneHotWidth = dataset.dataArray.shape
    z = cp.Variable(datasetLength, boolean=True) # subset decision vector
    t = cp.Variable(oneHotWidth) # L1 norm linearization vector
    ones = np.ones(oneHotWidth) # ones vector indicating every bin
    subsetCoverage = cp.Variable(oneHotWidth)
    dataset_coverage = np.minimum(ones, np.sum(dataset.dataArray, axis=0))

    # L1 norm linearization constraints and s constraint
    constraints = [cp.sum(z) == subsetSize,
                   subsetCoverage <= 1,
                   subsetCoverage <= z@dataset.dataArray,
                   -t <= dataset_coverage - subsetCoverage,
                   t >= dataset_coverage - subsetCoverage]

    objective = cp.Minimize(cp.sum(t)) # objective is maximizing the sum of t
    problem = optimize(objective=objective, 
                       constraints=constraints, 
                       environment=environment, 
                       verbose=verbose)

    time_1 = time.time()
    timeTotal = time_1 - time_0

    return z.value.astype(int), timeTotal, problem.value


def optimizeSum(dataset, environment, subsetSize, w, verbose=False):

    time_0 = time.time()

    datasetLength = len(dataset.dataArray)
    z = cp.Variable(datasetLength, boolean=True) # subset decision vector

    # L1 norm linearization constraints and s constraint
    constraints = []

    objective = cp.Minimize(w[0]*cp.sum(z) - w[1]*cp.sum(z@dataset.dataArray))
    problem = optimize(objective=objective, 
                       constraints=constraints, 
                       environment=environment, 
                       verbose=verbose)

    time_1 = time.time()
    timeTotal = time_1 - time_0

    return z.value.astype(int), timeTotal, problem.value
