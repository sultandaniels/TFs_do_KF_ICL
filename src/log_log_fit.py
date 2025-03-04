import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import sympy as sp

def model_function(x, a, b, c):
    return c + np.exp(b) * x**a

def model_function_loglin(x, a, b, c):
    x = np.array(x) # Convert to numpy array
    return c + np.exp(b) * np.exp(a*x)

def is_psd(matrix):
    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)

def closed_form_loglin_constants(x):
    n = len(x)
    x = np.array(x).reshape((n,1))

    if not is_psd(np.array([[(x.T @ x).item(), np.sum(x).item()], [np.sum(x).item(), n]])):
        raise ValueError("Optimization problem is not convex")

    det = 1/(n*x.T @ x - np.sum(x)**2)
    A = det * np.array([[n, -np.sum(x)],[-np.sum(x), np.sum(x**2)]])

    return A

def closed_form_loglin(x, y, c):
    n = len(x)
    x = np.array(x).reshape((n,1))
    y = np.array(y).reshape((n,1))
    u = np.abs(y - c*np.ones((n,1)))
    log_u = np.log(u)

    A = closed_form_loglin_constants(x)
    z = np.array([log_u.T @ x, log_u.T @ np.ones((n,1))]).flatten()

    a, b = A @ z

    diff = log_u - a*x - b*np.ones((n,1))
    err = diff.T @ diff

    diff_lin = u - np.exp(a*x + b*np.ones((n,1)))
    err_lin = diff_lin.T @ diff_lin
    return a,b, err, err_lin

def plot_closed_form_loglin_err(x, y, irreducible_error, ax, sys, t, min, max):
    amt = int(1e5)
    c_vals = np.linspace(min, max, amt)
    err_vals = np.zeros(amt)
    err_lin_vals = np.zeros(amt)
    a_vals = np.zeros(amt)
    b_vals = np.zeros(amt)

    for i in range(amt):
        a_vals[i], b_vals[i], err_vals[i], err_lin_vals[i] = closed_form_loglin(x, y, c_vals[i])

    # plt.plot(c_vals, err_vals, label="Error", marker='o')
    # plt.title("Log Error vs c")
    ax[sys].plot(c_vals, err_lin_vals, marker='.', label="t="+str(t))
    ax[sys].set_title("Error vs c for system " + str(sys) + ". Irreducible error: " + str(irreducible_error))
    ax[sys].set_xlabel("c")
    ax[sys].set_ylabel("Error")
    ax[sys].legend()
    return ax, a_vals, b_vals, c_vals, err_vals, err_lin_vals

# def find_c(x,y, irreducible_error,lr, stop_criteria=1e-3, max_iter=1000):
#     # Find c by gradient descent
#     n = len(x)
#     c = irreducible_error
#     for i in range(max_iter):
#         c_grad = 2/n * np.sum(y - c - np.exp(lr) * x**lr)
#         c = c - 0.01 * c_grad
#         if np.abs(c_grad) < stop_criteria:
#             break
#     return c

# Define the loss function with regularization on b
def loss(lambda_reg, x_values, y_values, params):
    a, b, c = params
    y_pred = model_function(x_values, a, b, c)
    # Regular least squares loss
    loss_value = np.sum((y_values - y_pred)**2)
    # Add regularization term for b
    loss_value += lambda_reg * b**2
    return loss_value

def loglogfit(x_train, x_values, y_train, initial_guess):
    try:
        # Use curve_fit to fit the model function to the data
        params, covariance = curve_fit(model_function, x_train, y_train, p0=initial_guess)
        
        # Extract the parameters
        a, b, c = params
        
        # Generate y-values based on the fitted model
        fitted_y_values = model_function(x_values, a, b, c)
    except RuntimeError:
        print("Optimal parameters not found: Number of calls to function has reached maxfev.")
        # Set default values
        a, b, c = initial_guess
        fitted_y_values = model_function(x_values, a, b, c)
    
    return fitted_y_values, a, b, c

def loglinfit(x_train, x_values, y_train, initial_guess):
    
    # Use curve_fit to fit the model function to the data
    params, covariance = curve_fit(model_function_loglin, x_train, y_train, p0=initial_guess)
    
    # Extract the parameters
    a, b, c = params
    
    # Generate y-values based on the fitted model
    fitted_y_values = model_function_loglin(x_values, a, b, c)
    
    return fitted_y_values, a, b, c

def loglogfit_linear(x_train, x_values, y_train):
    # For a log-log scale regression
    log_x = np.log(x_train)
    log_x_values = np.log(x_values)
    log_y = np.log(y_train)

    # Set up the design matrix for a linear model
    A = np.vstack([log_x, np.ones(len(log_x))]).T

    # Solve the least squares problem
    m, c = np.linalg.lstsq(A, log_y, rcond=None)[0]

    # To plot or use the regression line:
    # Convert back if you're working on a log-log scale
    predicted_y = np.exp(m * log_x_values + c)
    return predicted_y, m, c

def loglogfit_regularized(initial_guess, x_train, y_train, lambda_reg=0.01):
    ## regularized version
    # Initial guess for parameters

    # Perform the minimization
    result = minimize(lambda params: loss(lambda_reg, x_train, y_train, params), initial_guess)

    # Extract the optimized parameters
    a_opt, b_opt, c_opt = result.x
    return a_opt, b_opt, c_opt

if __name__ == '__main__':
    
    # # Define the parameters for the curve
    # a = -2.0  # Example value for a
    # b = -0.5 # Example value for b
    # c = 10   # Example value for c

    # # Define the range of x values
    # x_values = np.arange(1, 11)  # Generate integers from 1 to 10
    # x_values = x_values.astype(float)  # Convert to float

    # # Calculate y values using the curve equation
    # y_values = c + np.exp(b) * x_values**a + np.random.normal(0, 1e-3, len(x_values))

    # # Collect the ordered pairs
    # ordered_pairs = list(zip(x_values, y_values))

    # fitted_y_values, a_f, b_f, c_f = loglogfit(x_values, y_values)
    
    # # Plot the data and the regression line
    # plt.scatter(x_values, y_values-c_f, label="Data")
    # plt.plot(x_values, fitted_y_values-c_f, label="Fitted Curve", color="red")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.xlabel("x")
    # plt.ylabel("y")

    # lambda_reg = 0.01
    # initial_guess = [-1.0, 0.0, 1.0]
    # a_opt, b_opt, c_opt = loglogfit_regularized(initial_guess, x_values, y_values, lambda_reg)

    
    # print(f"Optimized parameters: a={a_opt}, b={b_opt}, c={c_opt}")
    # # Generate y-values based on the optimized model
    # fitted_y_values_opt = model_function(x_values, a_opt, b_opt, c_opt)
    # plt.plot(x_values, fitted_y_values_opt-c_opt, label="Regularized Fitted Curve", color="green")
    # plt.legend()


    # Sample data for exponential decay
    x_values = np.linspace(0, 10, 10)
    y_values = 5 * np.exp(-0.5 * x_values) + np.random.normal(0, 0.2, x_values.shape)

    # Fit the model
    fitted_y_values, a, b, c = loglinfit(x_values, y_values)

    # Plot the results
    plt.scatter(x_values, y_values, label='Data')
    plt.plot(x_values, fitted_y_values, label='Fitted Model', color='red')
    plt.legend()
    plt.show()

    plt.show()