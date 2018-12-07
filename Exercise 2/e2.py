"""
ELEC- E8739 AI in Health Technologies. Exercise II
Numerical Experiemt for CLT
Code by Zheng Zhao
Aalto University
Sep 14, 2018
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

def fx_sampling(T, a, b):
	"""
	Inverse sampling method for fx(x). The loop for T is very ugly, as it is only used for 
	demenstrating the basic idea for good understanding.
	In practice, you should generate u=np.random.rand(T) and operate on vector without any loop.
	"""
	c = (1-b + a*b) / a
	z = np.zeros(shape=(T,))
	for i in range(T):
		u = np.random.rand()
		if u<= a*c:
			z[i] = u/c
		else:
			z[i] = (u+a*b-a*c) / b
	return z

m = 2000 # How many samples of Z we want
n = 1000 # How many samples of X we want to sum into Z

a = 0.5   # Your desired a
b = 0.5   # Your desired b (can not > 1)

# Calculate c
c = (1-b + a*b) / a

# Show that our fx_sampling() work
plt.hist(fx_sampling(m, a, b), density=True)
plt.show()

# CLT
z = np.zeros((m,))
for i in range(m):
	z[i] = np.sum(fx_sampling(n, a, b))/n

# The theoretical expectation and variance
mu = 0.5*(c*a**2+b-b*a**2)
sigma = 1/3 * (c*((a-mu)**3+mu**3) + b*((1-mu)**3-(a-mu)**3))

z_ = np.sqrt(n) * (z - mu) # which should result in N(0, sigma^2)

# Plot histogram of z_ and plot the theoretical Normal PDF.
plt.hist(z_, density=True, label='Histogram from experiment')
x = np.linspace(0 - 10*sigma, 0 + 10*sigma, 1000)
plt.plot(x, norm.pdf(x, loc=0, scale=np.sqrt(sigma)), label='Theoretical result')
plt.legend()
plt.show()