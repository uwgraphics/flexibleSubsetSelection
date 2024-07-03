# --- Imports ------------------------------------------------------------------

# Standard library
import time

# Third party libraries
import cvxpy as cp
import numpy as np


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

def bestOfRandom(dataset, lossFunction, subsetSize, minLoss=0, 
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


def averageOfRandom(dataset, lossFunction, subsetSize, minLoss=0, 
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


def worstOfRandom(dataset, lossFunction, subsetSize, minLoss=0, 
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


def greedySwap(dataset, lossFunction, subsetSize, minLoss=0, 
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
              f"{lossFunction.objective.__name__} objective.")
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

def greedyMinSubset(dataset, lossFunction, epsilon, 
                    minError=0, maxIterations=None, seed=None, 
                    verbose=False, initialSize=1):
    """
    A greedy algorithm for subset selection to minimize the size of the subset 
    such that lossFunction(subset) <= epsilon.

    Args:
        dataset (object): The Dataset class object
        lossFunction (object): The loss function class object
        epsilon (float): The target value for the function
        solveArray (string): The desired array to use during solving
        minError (float): The minimum error value to stop iterations
        maxIterations (int, optional): Maximum number of iterations
        seed (int, rng, optional): The random seed or NumPy rng for random 
            generation and reproducibility
        verbose (bool, optional): Toggle for verbose logging
        initialSize (int, optional): Initial size of the subset

    Returns:
        z (array): Indicator vector of included items in the subset
        timeTotal (float): Total execution time
        error (float): The error value of the final subset
    """
    import time
    import numpy as np

    # Extract dataset size
    datasetLength = dataset.size[0]

    if verbose:
        print(f"Solving for a subset such that {lossFunction.objective.__name__}(subset) <= {epsilon}")
    time0 = time.time()  # get start time
    iterations = 0
    consecutive_stable_iterations = 0
    prev_subset_size = initialSize

    # Set the random seed
    np.random.seed(seed)
    
    # Initialize the indicator vector z
    z = np.zeros(datasetLength, dtype=int)
    
    # Randomly select initial points
    selected_indices = np.random.choice(datasetLength, initialSize, replace=False)
    z[selected_indices] = 1
    
    # Set of available indices
    available_indices = set(range(datasetLength))
    available_indices.difference_update(selected_indices)
    
    # Initial loss calculation
    current_loss = lossFunction.calculate(dataset, z)
    error = abs(current_loss - epsilon)

    if maxIterations is None:
        maxIterations = datasetLength

    while iterations < maxIterations:
        if verbose:
            print(f"Iteration {iterations}: Loss {current_loss}, Error {error}, Subset Size {np.sum(z)}")
        
        # Check if error is less than or equal to epsilon
        if error <= epsilon:
            # Attempt to drop one or more points while keeping error below epsilon
            dropped = False
            new_selected_indices = []  # Create a list to store new selected indices
            
            for index in selected_indices:
                if z[index] == 1:  # Ensure the index is currently selected
                    z[index] = 0
                    new_loss = lossFunction.calculate(dataset, z)
                    new_error = abs(new_loss - epsilon)
                    
                    if new_error <= epsilon:
                        current_loss = new_loss
                        error = new_error
                        available_indices.add(index)
                        dropped = True
                    else:
                        z[index] = 1  # Revert the change
                        new_selected_indices.append(index)  # Keep index in selected list
                else:
                    new_selected_indices.append(index)  # Keep index in selected list
            
            selected_indices = np.array(new_selected_indices)
            
            if dropped:
                consecutive_stable_iterations = 0  # Reset consecutive stable iterations
                prev_subset_size = np.sum(z)
                continue  # Continue optimizing if dropped points successfully
            else:
                consecutive_stable_iterations += 1

                # Check if subset size has not changed for a while
                if consecutive_stable_iterations >= 5:
                    break
        
        best_index = None
        best_error = error

        for index in available_indices:
            z[index] = 1  # try adding this element
            new_loss = lossFunction.calculate(dataset, z)
            new_error = abs(new_loss - epsilon)
            
            if new_error < best_error:
                best_error = new_error
                best_index = index
            
            z[index] = 0  # revert the addition

        if best_index is not None:
            z[best_index] = 1
            available_indices.remove(best_index)
            current_loss = lossFunction.calculate(dataset, z)
            error = abs(current_loss - epsilon)

        iterations += 1

    time1 = time.time()  # get end time
    timeTotal = time1 - time0  # calculate total time

    return z, timeTotal, error

import numpy as np
import time

def greedyMixed(dataset, lossFunction, weight=1.0, 
                            minError=0, maxIterations=None, seed=None, 
                            verbose=False, initialSize=1):
    """
    A greedy algorithm to minimize the total loss = weight * subsetSize + lossFunction.calculate().

    Args:
        dataset (object): The Dataset class object
        lossFunction (object): The loss function class object
        weight (float): Weight parameter for the subset size
        minError (float): The minimum error value to stop iterations
        maxIterations (int, optional): Maximum number of iterations
        seed (int, rng, optional): The random seed or NumPy rng for random 
            generation and reproducibility
        verbose (bool, optional): Toggle for verbose logging
        initialSize (int, optional): Initial size of the subset

    Returns:
        z (array): Indicator vector of included items in the subset
        timeTotal (float): Total execution time
        total_loss (float): The total loss value of the final subset
    """
    # Extract dataset size
    datasetLength = dataset.size[0]

    if verbose:
        print(f"Solving to minimize total loss = {weight} * subsetSize + lossFunction.calculate()")
    time0 = time.time()  # get start time
    iterations = 0

    # Set the random seed
    np.random.seed(seed)
    
    # Initialize the indicator vector z
    z = np.zeros(datasetLength, dtype=int)
    
    # Randomly select initial points
    selected_indices = np.random.choice(datasetLength, initialSize, replace=False)
    z[selected_indices] = 1
    
    # Set of available indices
    available_indices = set(range(datasetLength))
    available_indices.difference_update(selected_indices)
    
    # Initial loss calculation
    current_loss = lossFunction.calculate(dataset, z)
    total_loss = weight * np.sum(z) + current_loss
    error = abs(total_loss)

    if maxIterations is None:
        maxIterations = datasetLength

    while iterations < maxIterations:
        if verbose:
            print(f"Iteration {iterations}: Total Loss {total_loss}, Subset Size {np.sum(z)}")
        
        # Check if error is less than or equal to minError
        if error <= minError:
            break

        best_index = None
        best_total_loss = total_loss

        for index in available_indices:
            z[index] = 1  # try adding this element
            new_loss = lossFunction.calculate(dataset, z)
            new_total_loss = weight * np.sum(z) + new_loss
            

            if new_total_loss < best_total_loss:
                best_total_loss = new_total_loss
                best_index = index
            
            z[index] = 0  # revert the addition

        if best_index is not None:
            z[best_index] = 1
            available_indices.remove(best_index)
            current_loss = lossFunction.calculate(dataset, z)
            total_loss = weight * np.sum(z) + current_loss
            error = abs(total_loss)  # update error
        
        iterations += 1

    time1 = time.time()  # get end time
    timeTotal = time1 - time0  # calculate total time

    return z, timeTotal, total_loss  # return indicator vector, total time, and total loss


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
