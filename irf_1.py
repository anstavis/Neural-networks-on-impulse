import numpy as np
import matplotlib.pyplot as plt

# Coefficient matrices from the VAR(2) summary
A1 = np.array([
    [-0.319631, 0.145989, 0.961219],
    [0.043931, -0.152732, 0.288502],
    [-0.002423, 0.224813, -0.263968]
])

A2 = np.array([
    [-0.160551, 0.114605, 0.934394],
    [0.050031, 0.019166, -0.010205],
    [0.033880, 0.354912, -0.022230]
])

# Identity matrix for Phi_0
Phi_0 = np.eye(3)

# Initialize list of Phi matrices
Phi = [Phi_0, A1]

# Recursive calculation of Phi matrices (up to 8 periods)
for i in range(2, 8):
    Phi_i = A1 @ Phi[i-1] + A2 @ Phi[i-2]
    Phi.append(Phi_i)

# Shock vector: shock to the "income" variable (second variable)
shock = np.array([0, 1, 0])

# Calculate the IRFs for "cons" (third variable) in response to the shock
irf_cons = [Phi_i @ shock for Phi_i in Phi]
irf_cons_values = [irf[2] for irf in irf_cons]  # Extract the third variable response

print(irf_cons_values)

# Plotting the IRF values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 9), irf_cons_values, marker='o', linestyle='-', color='b')
plt.title("Impulse Response Function of 'cons' to a Shock in 'income'")
plt.xlabel("Time Steps")
plt.ylabel("Response of 'cons'")
plt.grid(True)
plt.show()