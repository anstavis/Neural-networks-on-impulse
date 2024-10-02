import numpy as np
import matplotlib.pyplot as plt

# Parameters
s = 3
x0 = 0.2
e = -1
h = 7
N = 10000

# Generate innovation values
eps = 0.1 * np.random.randn(s + h, N)

# Set shocks to zero for t < s
eps[:s - 1, :] = 0
eps[s - 1, :] = e

# Initialize x with zeros and set initial conditions
x = np.zeros((s + h, N))
x[:s - 1, :] = x0 * np.ones((s - 1, N))
x[s - 1, :] = (x0 + e) * np.ones(N)
print(len(x))


# Simulate values recursively
for t in range(s, s + h):
    x[t, :] = np.tanh(0.9 * x[t - 1, :]) + eps[t, :]

# Obtain IRF recursively
x_tilde = np.mean(x, axis=1)
upper_bound = np.percentile(x, 95, axis=1)
lower_bound = np.percentile(x, 5, axis=1)

# Plot the results
plt.plot(upper_bound, 'r', label='Upper bound (95th percentile)')
plt.plot(lower_bound, 'r', label='Lower bound (5th percentile)')
plt.plot(x_tilde, 'k', label='IRF (mean)')
plt.legend()
plt.show()
