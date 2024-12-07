import numpy as np
import statsmodels
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

def get_empirical_cdf(values):
    # Ensure the input is a numpy array
    values = np.array(values)
    
    # Compute the empirical CDF
    ecdf = ECDF(values)
    
    return ecdf

if __name__ == '__main__':
    # Example usage
    values = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]
    ecdf = get_empirical_cdf(values)

    # Plot the empirical CDF
    plt.step(ecdf.x, ecdf.y, where='post')
    plt.xlabel('Values')
    plt.ylabel('ECDF')
    plt.title('Empirical CDF')
    plt.grid(True)
    plt.show()