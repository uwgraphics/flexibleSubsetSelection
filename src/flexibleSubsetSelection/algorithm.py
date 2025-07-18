# --- Imports ------------------------------------------------------------------

import math

# Third party
import cvxpy as cp
import gurobipy as gp
import numpy as np
import ot

# Local files
from . import logger
from .loss import UniCriterion, MultiCriterion

# Setup logger
log = logger.setup(__name__)


# --- Utility ------------------------------------------------------------------


def randomSample(
    datasetSize: tuple, subsetSize: int, seed: (int | np.random.Generator | None) = None
):
    """
    Randomly sample from dataset by generating random indices to create subset

    Args:
        datasetSize: The dataset dimensions sizes
        subsetSize: The number of points to sample for the subset
        seed: The random seed or generator for reproducibility.

    Returns:
        z (array): indicator vector of included items in the subset
        indices (array): indices in the dataset of included items in the subset
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(datasetSize[0], size=subsetSize, replace=False)
    z = np.zeros(datasetSize[0], dtype=bool)
    z[indices] = True
    return z, indices


def createEnvironment(outputFlag: int = 0):
    """
    Create and set up an environment required by the Gurobi solver

    Arg: outputFlag: Flag for Gurobi output.
    """
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", outputFlag)
    environment.setParam("LogFile", "../data/gurobiLog.log")
    # environment.setParam("ConcurrentMIP", 2)
    environment.start()

    return environment


def optimize(objective, constraints, environment, solver, log_file="gurobi_log.txt"):
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
            problem.
        log_file: Optional. File path for Gurobi log. Defaults to 'gurobi_log.txt'.

    Returns: problem: The cvxpy Problem object after solving, which contains
        solution information and other attributes.
    """
    problem = cp.Problem(objective, constraints)

    if solver == cp.GUROBI:
        problem.solve(solver=solver, env=environment, logfile=log_file)
    else:
        problem.solve(solver=solver)

    return problem


# --- Algorithms ---------------------------------------------------------------


def bestOfRandom(
    dataset,
    lossFunction: (UniCriterion | MultiCriterion),
    subsetSize: int,
    minLoss: int = 0,
    maxIterations: int = None,
    seed: (int | np.random.Generator | None) = None,
    selectBy: str = "row",
) -> tuple[np.ndarray, float]:
    """
    Determine the best loss of subset selection of n random samples
    """
    rng = np.random.default_rng(seed)
    if maxIterations is None:
        maxIterations = dataset.size[0]

    z = randomSample(dataset.size, subsetSize, rng)[0]
    minLoss = lossFunction(dataset, z)

    for i in range(maxIterations):
        log.debug("%s: %s", i, minLoss)
        curZ = randomSample(dataset.size, subsetSize, rng)[0]
        curLoss = lossFunction(dataset, curZ)
        if curLoss < minLoss:
            z = curZ
            minLoss = curLoss

    return z, minLoss


def averageOfRandom(
    dataset,
    lossFunction: (UniCriterion | MultiCriterion),
    subsetSize: int,
    minLoss: int = 0,
    maxIterations: int = None,
    seed: (int | np.random.Generator | None) = None,
    selectBy: str = "row",
) -> tuple[np.ndarray, float]:
    """
    Determine the average loss of subset selection of n random samples
    """
    rng = np.random.default_rng(seed)
    if maxIterations is None:
        maxIterations = dataset.size[0]

    z = randomSample(dataset.size, subsetSize, rng)[0]
    losses = [lossFunction(dataset, z)]

    for i in range(maxIterations):
        curZ = randomSample(dataset.size, subsetSize, rng)[0]
        losses.append(lossFunction(dataset, curZ))

    avgLoss = np.mean(losses)

    return z, avgLoss


def worstOfRandom(
    dataset,
    lossFunction: (UniCriterion | MultiCriterion),
    subsetSize: int,
    minLoss: int = 0,
    maxIterations: int = None,
    seed: (int | np.random.Generator | None) = None,
    selectBy: str = "row",
) -> tuple[np.ndarray, float]:
    """
    Determine the worst loss of subset selection of n random samples
    """
    rng = np.random.default_rng(seed)
    if maxIterations is None:
        maxIterations = dataset.size[0]

    z = randomSample(dataset.size, subsetSize, rng)[0]
    maxLoss = lossFunction(dataset, z)

    for i in range(maxIterations):
        curZ = randomSample(dataset.size, subsetSize, rng)[0]
        curLoss = lossFunction(dataset, curZ)
        if curLoss > maxLoss:
            z = curZ
            maxLoss = curLoss

    return z, maxLoss


def greedySwap(
    dataset,
    lossFunction,
    subsetSize,
    minLoss=0,
    maxIterations=None,
    seed=None,
    callback=None,
):
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

    Returns:
        z (array): Indicator vector of included items in the subset
        loss (float): The loss value of the final subset
    """
    rng = np.random.default_rng(seed)
    log.debug("Solving for a subset of size %s.", subsetSize)
    iterations = 0

    # select random starting subset
    z, indices = randomSample(dataset.size, subsetSize, rng)
    loss = lossFunction(dataset, z)

    if maxIterations is None:
        maxIterations = dataset.size[0]

    for i in range(maxIterations):
        log.debug("Iteration %s/%s: Loss %s.", i, maxIterations, loss)
        if callback:
            callback(iterations, loss)

        if i not in indices:
            zSwapBest = np.copy(z)
            lossSwapBest = loss
            indicesSwapBest = np.copy(indices)

            for loc, j in enumerate(indices):
                iterations += 1
                zSwap = np.copy(z)
                zSwap[i] = 1  # add the new datapoint
                zSwap[j] = 0  # remove the old datapoint
                indicesSwap = np.copy(indices)
                indicesSwap[loc] = i
                lossSwap = lossFunction(dataset, zSwap)

                if lossSwap < lossSwapBest:
                    zSwapBest = np.copy(zSwap)
                    lossSwapBest = lossSwap
                    indicesSwapBest = indicesSwap

            z = np.copy(zSwapBest)
            loss = lossSwapBest
            indices = indicesSwapBest

            if loss == minLoss:
                break

    return z, loss  # return indicator and final loss

def greedySwapG(
    dataset,
    lossFunction,
    subsetSize,
    minLoss: float = 0,
    maxIterations: int | None = None,
    swapRange: tuple[int, int] = (1, 20),  # (min, max) number of swaps per iteration
    seed=None,
    callback=None,
):
    """
    Greedy multi-swap subset selection algorithm with randomized swap counts per iteration.
    """
    rng = np.random.default_rng(seed)
    log.debug("Solving for a subset of size %s.", subsetSize)

    z, _ = randomSample(dataset.size, subsetSize, rng)
    currentLoss = lossFunction(dataset, z)
    iterations = 0

    if maxIterations is None:
        maxIterations = 10 * subsetSize

    for iter_ in range(maxIterations):
        if callback:
            callback(iterations, currentLoss)
        log.debug("Iter %d: loss %.6f", iter_, currentLoss)

        num_swaps = rng.integers(*swapRange)
        in_idx = np.where(z)[0]
        out_idx = np.where(~z)[0]

        if len(in_idx) < num_swaps or len(out_idx) < num_swaps:
            log.debug("Not enough elements to swap — stopping.")
            break

        # Propose multi-swap
        idxs_out = rng.choice(in_idx, size=num_swaps, replace=False)
        idxs_in = rng.choice(out_idx, size=num_swaps, replace=False)

        z_new = z.copy()
        z_new[idxs_out] = False
        z_new[idxs_in] = True

        newLoss = lossFunction(dataset, z_new)
        iterations += 1

        if newLoss < currentLoss:
            z = z_new
            currentLoss = newLoss
            log.debug("  accepted %d swaps → loss %.6f", num_swaps, currentLoss)
            if currentLoss <= minLoss:
                log.debug("Reached minLoss %.6f — terminating.", currentLoss)
                break
        else:
            log.debug("  rejected %d swaps", num_swaps)

    return z, currentLoss

def simulatedAnnealing(
    dataset,
    lossFunction,
    subsetSize: int,
    *,
    Tmax: float = 1.0,          # initial temperature
    Tmin: float = 1e-3,         # final temperature
    cooling: float = 0.9,      # geometric cooling factor (0<T<1)
    steps_per_T: int = 50,     # how many proposals per temperature level
    swap_range: tuple[int,int] = (1, 5000),  # min / max points swapped per proposal
    seed=None,
    callback=None,
):
    """
    Simulated-annealing subset selection.

    Parameters
    ----------
    dataset       : your Dataset object
    lossFunction  : callable(dataset, z) -> float
    subsetSize    : desired subset cardinality
    Tmax, Tmin    : start / end temperature
    cooling       : geometric cooling factor (e.g. 0.95)
    steps_per_T   : proposals evaluated at each temperature
    swap_range    : (min,max) number of indices swapped in one proposal
    seed          : RNG seed / Generator
    callback      : optional f(iterations, loss, T) for live plotting
    """
    rng = np.random.default_rng(seed)

    # --- initial subset ------------------------------------------------------
    z, _ = randomSample(dataset.size, subsetSize, rng)
    best_z  = z.copy()
    curLoss = bestLoss = lossFunction(dataset, z)
    iterations = 0
    T = Tmax

    in_idx  = np.where(z)[0]
    out_idx = np.where(~z)[0]

    log.info("SA start: loss %.6f", curLoss)

    # --- annealing loop ------------------------------------------------------
    while T > Tmin:
        for _ in range(steps_per_T):
            # -- propose a random multi-swap ----------------------------
            k = rng.integers(*swap_range)                 # how many to swap
            if k > len(in_idx) or k > len(out_idx):
                k = min(len(in_idx), len(out_idx))
                if k == 0:
                    break

            idx_out = rng.choice(in_idx,  size=k, replace=False)
            idx_in  = rng.choice(out_idx, size=k, replace=False)

            z_new = z.copy()
            z_new[idx_out] = False
            z_new[idx_in]  = True
            newLoss = lossFunction(dataset, z_new)
            iterations += 1

            # -- Metropolis acceptance test ----------------------------
            dE = newLoss - curLoss
            accept = (dE < 0) or (rng.random() < math.exp(-dE / T))

            if accept:
                z        = z_new
                curLoss  = newLoss
                in_idx   = np.where(z)[0]
                out_idx  = np.where(~z)[0]

                if curLoss < bestLoss:
                    bestLoss = curLoss
                    best_z   = z.copy()

                if callback:
                    callback(iterations, curLoss)

                log.debug("T=%.4g  best=%.6f  current=%.6f", T, bestLoss, curLoss)

        T *= cooling  # geometric cooling

    log.info("SA done: best loss %.6f after %d loss evaluations", bestLoss, iterations)
    return best_z, bestLoss


def greedyMinSubset(
    dataset,
    lossFunction,
    epsilon,
    minError=0,
    maxIterations=None,
    seed=None,
    initialSize=1,
    callback=None,
):
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
        initialSize (int, optional): Initial size of the subset
        callback (function, optional): A callback function for loss values

    Returns:
        z (array): Indicator vector of included items in the subset
        timeTotal (float): Total execution time
        error (float): The error value of the final subset
    """

    # Extract dataset size
    datasetLength = dataset.size[0]

    log.debug("Solving for a subset such that loss(subset) <= %s.", epsilon)
    iterations = 0
    consecutive_stable_iterations = 0

    # Set the random seed
    rng = np.random.default_rng(seed)

    # Initialize the indicator vector z
    z = np.zeros(datasetLength, dtype=int)

    # Randomly select initial points
    z, selected_indices = randomSample(dataset.size, initialSize, rng)

    # Set of available indices
    available_indices = set(range(datasetLength))
    available_indices.difference_update(selected_indices)

    # Initial loss calculation
    current_loss = lossFunction(dataset, z)
    error = abs(current_loss - epsilon)

    if maxIterations is None:
        maxIterations = datasetLength

    while iterations < maxIterations:
        log.debug(
            "Iteration: %s, Loss: %s, Error: %s, Subset Size: %s.",
            iterations,
            current_loss,
            error,
            np.sum(z),
        )
        if callback:
            callback(iterations, current_loss, subsetSize=np.sum(z))

        # Check if error is less than or equal to epsilon
        if error <= epsilon:
            # Attempt to drop one or more points while keeping error below epsilon
            dropped = False
            new_selected_indices = []  # Create a list to store new selected indices

            for index in selected_indices:
                if z[index] == 1:  # Ensure the index is currently selected
                    z[index] = 0
                    new_loss = lossFunction(dataset, z)
                    new_error = abs(new_loss - epsilon)

                    if new_error <= epsilon:
                        current_loss = new_loss
                        error = new_error
                        available_indices.add(index)
                        dropped = True
                    else:
                        z[index] = 1  # Revert the change
                        new_selected_indices.append(
                            index
                        )  # Keep index in selected list
                else:
                    new_selected_indices.append(index)  # Keep index in selected list

            selected_indices = np.array(new_selected_indices)

            if dropped:
                consecutive_stable_iterations = 0  # Reset consecutive stable iterations
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
            new_loss = lossFunction(dataset, z)
            new_error = abs(new_loss - epsilon)

            if new_error < best_error:
                best_error = new_error
                best_index = index

            z[index] = 0  # revert the addition

        if best_index is not None:
            z[best_index] = 1
            available_indices.remove(best_index)
            current_loss = lossFunction(dataset, z)
            error = abs(current_loss - epsilon)

        iterations += 1

    return z, error


def greedyMixed(
    dataset,
    lossFunction,
    weight=1.0,
    minError=0,
    maxIterations=None,
    seed=None,
    initialSize=1,
    callback=None,
):
    """
    A greedy algorithm to minimize the total
    loss = weight * subsetSize + lossFunction().

    Args:
        dataset (object): The Dataset class object
        lossFunction (object): The loss function class object
        weight (float): Weight parameter for the subset size
        minError (float): The minimum error value to stop iterations
        maxIterations (int, optional): Maximum number of iterations
        seed (int, rng, optional): The random seed or NumPy rng for random
            generation and reproducibility
        initialSize (int, optional): Initial size of the subset

    Returns:
        z (array): Indicator vector of included items in the subset
        total_loss (float): The total loss value of the final subset
    """
    # Extract dataset size
    datasetLength = dataset.size[0]

    log.debug(
        "Solving to minimize total loss = %s * subsetSize + lossFunction()",
        weight,
    )
    iterations = 0
    rng = np.random.default_rng(seed)

    # Initialize the indicator vector z
    z = np.zeros(datasetLength, dtype=int)

    # Randomly select initial points
    z, selected_indices = randomSample(dataset.size, initialSize, rng)

    # Set of available indices
    available_indices = set(range(datasetLength))
    available_indices.difference_update(selected_indices)

    # Initial loss calculation
    current_loss = lossFunction(dataset, z)
    total_loss = weight * np.sum(z) + current_loss
    error = abs(total_loss)

    if maxIterations is None:
        maxIterations = datasetLength

    while iterations < maxIterations:
        log.debug(
            "Iteration %s: Total Loss %s, Subset Size %s",
            iterations,
            total_loss,
            np.sum(z),
        )
        if callback:
            callback(iterations, total_loss, subsetSize=np.sum(z))

        # Check if error is less than or equal to minError
        if error <= minError:
            break

        best_index = None
        best_total_loss = total_loss

        for index in available_indices:
            z[index] = 1  # try adding this element
            new_loss = lossFunction(dataset, z)
            new_total_loss = weight * np.sum(z) + new_loss

            if new_total_loss < best_total_loss:
                best_total_loss = new_total_loss
                best_index = index

            z[index] = 0  # revert the addition

        if best_index is not None:
            z[best_index] = 1
            available_indices.remove(best_index)
            current_loss = lossFunction(dataset, z)
            total_loss = weight * np.sum(z) + current_loss
            error = abs(total_loss)  # update error
        else:
            break

        iterations += 1

    return z, total_loss  # return indicator vector, and total loss


def optimizeCoverage(dataset, lossFunction, environment, subsetSize):
    """
    Optimize subset selection for coverage while minimizing L1 norm.

    Args:
        environment: The environment or solver settings for optimization.
        datasetOnehot (numpy.ndarray): The one-hot encoded dataset.
        subsetSize (int): The desired size of the subset.

    Returns:
        z (numpy.ndarray): Binary array indicating the selected subset.
        problem.value (float): The value of the optimization problem.
    """
    datasetLength, oneHotWidth = dataset.dataArray.shape
    z = cp.Variable(datasetLength, boolean=True)  # subset decision vector
    t = cp.Variable(oneHotWidth)  # L1 norm linearization vector
    ones = np.ones(oneHotWidth)  # ones vector indicating every bin
    subsetCoverage = cp.Variable(oneHotWidth)
    dataset_coverage = np.minimum(ones, np.sum(dataset.dataArray, axis=0))

    # L1 norm linearization constraints and s constraint
    constraints = [
        cp.sum(z) == subsetSize,
        subsetCoverage <= 1,
        subsetCoverage <= z @ dataset.dataArray,
        -t <= dataset_coverage - subsetCoverage,
        t >= dataset_coverage - subsetCoverage,
    ]

    objective = cp.Minimize(cp.sum(t))  # objective is maximizing the sum of t
    problem = optimize(
        objective=objective, constraints=constraints, environment=environment
    )

    return z.value.astype(int), problem.value


def optimizeSum(dataset, lossFunction, environment, w, solver):
    datasetLength = len(dataset.dataArray)
    z = cp.Variable(datasetLength, boolean=True)  # subset decision vector
    constraints = []

    objective = cp.Maximize(-w[0] * cp.sum(z) + w[1] * cp.sum(z @ dataset.dataArray))
    problem = optimize(objective, constraints, environment, solver)

    return z.value.astype(int), problem.value


def optimizeEMD(dataset, lossFunction, environment, subsetSize, solver=cp.GUROBI):
    datasetLength = len(dataset.dataArray)
    z = cp.Variable(datasetLength, boolean=True)  # subset decision vector
    constraints = [cp.sum(z) == subsetSize]
    subset = np.array(z @ dataset.dataArray)

    objective = cp.Minimize(ot.emd2([], [], ot.dist(subset, dataset.dataArray)))
    problem = optimize(objective, constraints, environment, solver)

    return z.value.astype(int), problem.value


def optimizeDistribution(dataset, lossFunction, environment, subsetSize):
    datasetLength, oneHotWidth = dataset.dataArray.shape
    z = cp.Variable(datasetLength, boolean=True)  # subset decision vector
    t = cp.Variable(oneHotWidth)  # L1 norm linearization vector

    oneHotMeans = np.sum(dataset.dataArray, axis=0) / datasetLength
    subsetMeans = (z @ dataset.dataArray) / subsetSize

    constraints = [
        cp.sum(z) == subsetSize,
        t >= 0,
        -t <= subsetMeans - oneHotMeans,
        t >= subsetMeans - oneHotMeans,
    ]

    objective = cp.Minimize(cp.sum(t))
    problem = optimize(objective, constraints, environment, solver=cp.GUROBI)

    return z.value.astype(int), problem.value


def sinkhorn(
    dataset, lossFunction, distanceMatrix, subsetSize, environment, lambdaReg=0.1
):
    datasetLength = dataset.size[0]

    # Decision variables
    z = cp.Variable(datasetLength, boolean=True)  # Subset selection vector
    gamma = cp.Variable((datasetLength, datasetLength), nonneg=True)

    # Minimize the Sinkhorn distance using the precomputed distance matrix
    objective = cp.Minimize(cp.sum(cp.multiply(gamma, distanceMatrix)))

    # Constraints
    constraints = [
        cp.sum(z) == subsetSize,
        cp.sum(gamma, axis=0) == 1 / datasetLength,
        cp.sum(gamma, axis=1) == z / subsetSize,
        gamma >= 0,
    ]

    # Formulate the problem
    problem = optimize(objective, constraints, environment, solver=cp.GUROBI)

    return z.value.astype(int), problem.value
