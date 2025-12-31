"""
Computes the maximal Lyapunov exponent for a ring of N coupled logistic maps,
and estimates the synchronous manifold domain.
"""

import numpy as np
import os

# ------- Parameters -------

r = 3.75
N = 8
eps_vals = np.linspace(0, 1, 501)

near_zero = 1e-12
trials = 30
updates = 1000
transients = 500
seed0 = 1

OUTPUT_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------- Functions --------

def logistic_map_update(x, r):
    # Logistic map: f(z) = rz(1-z)
    return r * x * (1.0 - x)
def CML_update(prev_state, r, coupling_matrix):
    """
    Updates the coupled logistic maps by one time-step:
    x_{i,t+1} = coupling_matrix · \mathbf f(\mathbf x)
    
    prev_state: N-dimensional vector
    r: scalar
    coupling_matrix: N^2-dimensional matrix
    """
    local = logistic_map_update(prev_state, r)
    updated_state = coupling_matrix.dot(local)
    return updated_state
def random_vector(dimensionality, magnitude, rng):
    # Returns a random unit vector of a scaled magnitude
    direction = rng.normal(size=dimensionality)
    direction /= np.linalg.norm(direction)
    return direction * magnitude
def Jacobian_matrix(state, r, coupling_matrix):
    """
    Returns the Jacobian (a N^2-dimensional matrix) evaluated at a certain state vector
    
    state: N-dimensional vector
    r: scalar
    coupling_matrix: N^2-dimensional matrix
    """
    W = r * (1.0 - 2.0 * state)
    J = coupling_matrix * W
    return J
def equal_ringed_coupling(epsilon):
    """
    Returns a N^2-dimensional matrix for the coupling.
    The coupling is ring shaped and nearest-neighbour:
    - diagonal: 1 - epsilon
    - neighbours of diagonal: epsilon/2
    """

    coupling = np.zeros((N, N), dtype=float)
    for i in range(N):
        left = (i - 1) % N
        right = (i + 1) % N
        neighbours = sorted(set([left, right]) - {i})

        if len(neighbours) == 0:
            coupling[i, i] = 1.0
        else:
            coupling[i, i] = 1.0 - epsilon
            share = epsilon / len(neighbours)
            for j in neighbours:
                coupling[i, j] = share

    return coupling
def MLE_estimate(N, coupling_matrix, seed):
    # Runs a single trial estimate of the maximal Lyapunov exponent
    rng = np.random.default_rng(seed)
    x = rng.uniform(1e-8, 1.0, size=N)
    delta = random_vector(dimensionality=N, magnitude=1, rng=rng)
    MLE = 0.0

    for i in range(transients):
        x = CML_update(x, r, coupling_matrix)

    for i in range(updates):
        delta = Jacobian_matrix(x, r, coupling_matrix).dot(delta)
        d = np.linalg.norm(delta)
        if d < near_zero:
            MLE += np.log(near_zero)
            delta = rng.normal(size=N)
            delta /= np.linalg.norm(delta)
        else:
            MLE += np.log(d)
            delta /= d
        x = CML_update(x, r, coupling_matrix)

    MLE_estimation = MLE / updates
    return MLE_estimation
def MLE_multiple_estimates(N, coupling_matrix):
    """
    Runs trials of maximal Lyapunov exponent estimation.
    Returns the mean and standard error of the mean of the trials.
    """
    estimates = []
    seed = seed0
    for i in range(trials):
        estimate = MLE_estimate(N, coupling_matrix, seed)
        estimates.append(estimate)
        seed += 1
    return np.mean(estimates), np.std(estimates) / np.sqrt(trials)
def synchronous_manifold_domain():
    """
    Estimates the analytic domain of the synchronous manifold using the numerical single logistic map MLE.
    For \epsilon \in (lo, hi), the synchronous state is stable.
    """
    MLE_single = MLE_multiple_estimates(N=1, coupling_matrix=np.array([1.0]))[0]
    
    delta = np.exp(-MLE_single)
    hi = 1.0
    lo = 0.0

    for k in range(1, N):
        c = np.cos((2 * np.pi * k) / N)

        if (1.0 - c) > near_zero:
            hi = min(hi, (1 + delta)/(1 - c))
            lo = max(lo, (1 - delta)/(1 - c))
        else:
            if delta < (1.0 - near_zero):
                return 1.0, 0.0
            else:
                continue
    return lo, hi

# ------- Main -------------

if __name__ == "__main__":
    results = []

    for eps in eps_vals:
        ans = MLE_multiple_estimates(N=N, coupling_matrix=equal_ringed_coupling(epsilon=eps))
        results.append([eps, ans[0], ans[1]])

    res = np.array(results, dtype=float)
    means = res[:, 1]
    sems  = res[:, 2]

    lo, hi = synchronous_manifold_domain()

    outfile = os.path.join(
        OUTPUT_DIR,
        f"MLE_fixedr{r}_N{N}_trials{trials}_updates{updates}.txt"
    )

    with open(outfile, "w") as f:
        f.write(f"Parameters:\nr={r}\nN={N}\ntrials={trials}updates={updates}transients={transients}")
        f.write(f"Synchronous domain: {lo} ≤ epsilon ≤ {hi}\n")
        f.write("Epsilon values\n")
        f.write(", ".join(map(str, eps_vals)) + "\n\n")
        f.write("MLE means\n")
        f.write(", ".join(map(str, means)) + "\n\n")
        f.write("MLE SEMs\n")
        f.write(", ".join(map(str, sems)) + "\n")
