"""
ELEC- E8739 AI in Health Technologies. Exercise II
Numerical Experiemt for CLT

The example given to you below demonstrates for a uniform distribution u(0,1).
The expected theoretical results should be: sqrt(n)(Z-mu) ~ N(0, sigma^2)
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

def fx_sampling(a,b,T):
        fx = np.zeros((T,))
        c = (1-b+a*b)/a
        x_ = np.random.rand(T)
        for i in range(T):
                x = x_[i]
                if x < a:
                        fx[i] = c
                else:
                        fx[i] = b
        return fx
        #return np.random.rand(T)

m = 2000 # How many Z sampls we want
n = 1000 # How many X samples we want to sum in Z

z = np.zeros((m,))
for i in range(m):
	z[i] = np.sum(fx_sampling(0.5, 0.5, n))/n
        #z[i] = np.sum(fx_sampling(n))/n

z_ = np.sqrt(n) * (z - np.mean(z)) # which should result in N(0, sigma^2)

# The theoretical results
sigma = np.std(z_)
mu = 0


# Plot histogram of z_ and plot the theoretical Normal PDF. 
plt.hist(z_, normed=True, label='Histogram from experiment')
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
plt.plot(x,norm.pdf(x, mu, sigma), label='Theoretical result')
plt.legend()
plt.show()
