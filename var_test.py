import numpy as np
from IPython.display import display
import pandas as pd
#Make var stays the same
def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def simulate_var(p, T, lag, sparsity=0.2, beta_range=(-0.3, 0.3), sd=0.1, seed=0, zeroing_prob=0.5):
    if seed is not None:
        np.random.seed(seed)

    # Set up Granger causality ground truth.
    GC = np.eye(p, dtype=int)

    # Generate the beta matrix for the VAR process (with lags)
    beta = np.zeros((p, p * lag))

    for i in range(p):
        # Ensure self-dependency for all lags
        for j in range(lag):
            beta[i, i + j * p] = np.random.uniform(beta_range[0], beta_range[1])  # Self-interaction
        
        # Select other random variables that influence variable i
        num_nonzero = int(p * sparsity)  # This determines how many other variables influence i
        if num_nonzero > 0:
            choice = np.random.choice([x for x in range(p) if x != i], size=num_nonzero, replace=False)
            for j in range(lag):
                # Randomly decide whether to zero out the coefficient
                if np.random.rand() > zeroing_prob:  # Keep with probability (1 - zeroing_prob)
                    beta[i, choice + j * p] = np.random.uniform(beta_range[0], beta_range[1], size=num_nonzero)
                    GC[i, choice] = 1  # Update Granger causality matrix

    print("Initial Beta:")
    beta_df = pd.DataFrame(beta)
    #display(beta_df)
    beta = make_var_stationary(beta)
    print("Stationary Beta:")
    beta_df = pd.DataFrame(beta)
    display(beta_df)

    # Generate data
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t]

    return X.T[burn_in:], beta, GC

# Simulate a VAR(3) process with 3 variables
p = 5  # Number of variables
T = 100  # Length of the time series
lag = 3  # VAR(3)
sparsity = 0.5  # 50% sparsity 
beta_range = (-0.3, 0.3)  # Random coefficients between -0.3 and 0.3
zeroing_prob = 0.5  # Probability of zeroing out a coefficient (excluding self-interaction)

# Simulate data
X_np, beta, GC = simulate_var(p, T, lag, sparsity, beta_range, zeroing_prob=zeroing_prob)

print("With the corresponding Granger Caqusality:")
display(GC)