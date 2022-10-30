from turtle import shape
import numpy as np
import matplotlib.pyplot as plt

# -----------------COMMON REFERENCE-----------------

# GENERAL IMPORTANT NOTES
# ndarrays are passed by reference.
# Changing an ndarray inside a function will change the original value passed.

# Initializing ----------
#   np.array(*ndarray*) initializes an ndarray
#   np.arange(*start*, *stop*, *step*) initializes an ndarray from a range, starting from *start* and ending exclusively at *stop*, with values separated by *step*
#   np.ones((*x dimension*, *y dimension*)) initializes an ndarray of ones
#   np.zeros((*x dimension*, *y dimension*)) initializes an ndarray of zeros


# Accessing / Adding / Changing Rows and Columns ----------  
#   *ndarray*[:,*column index*] returns a row vector from a matrix column
#   *ndarray*[:,[*column index*]] returns a column vector from a matrix column
#   *ndarray*[*row index*,:] returns a row vector from a matrix row
#   *ndarray*[*row index*,:].reshape(-1, 1)] returns a column vector from a matrix row
#   np.array([*1D array*]) returns a 2D array from a given 1D array (useful for transposing and other operations)
#
#   np.insert / np.append / np.hstack / np.vstack: adding on columns/rows
#   *ndarray*.shape returns a tuple of (#rows, #columns)
#       -*ndarray*.size() returns total number of elements
#       -len(*ndarray*) returns total number of "second highest dimension" sets, usually rows; using len(*ndarray*[0]) can get columns
#       -*ndarray*.ndim returns number of dimensions in the matrix


# Math ----------
# *ndarray1* @ *ndarray2* returns the matrix multiplication product of the two matrices.
# *ndarray*.T returns the transpose of the matrix


# Utility ----------
#   Reshaping:
#       *ndarray*.flatten(*optional character*) flattens/unrolls a matrix into a row vector, according to a scheme specified by the parameter character
#       *ndarray*.reshape((*x dimension*, *y dimension*)) ndarray method reshapes the matrix into the dimensions specified, which do not necessarily have to be in parentheses.
#           -Optional character C also allows for tweaking reshape scheme
#           -Pass -1 as a dimension to let numpy figure out the correct dimension
#       np.reshape(*ndarray*, (*x dimension*, *y dimension*)) numpy method reshapes the given matrix into the dimensions specified, which have to be in parentheses
#           -Optional character C also allows for tweaking reshape scheme
#           -Pass -1 as a dimension to let numpy figure out the correct dimension
#   Miscellaneous:
#       np.random.randint(*lower bound*, *upper bound*) returns a random integer from the lower bound (inclusive) to the upper bound (exclusive)
#       np.random.rand() returns a random decimal on the bound [0,1)




def prepare_data(coordinates: np.ndarray):

    X = coordinates[:,[0]]
    y = coordinates[:,[1]]
    return (X, y) 


def mse_cost(X:np.ndarray, y:np.ndarray, theta:np.ndarray):

    # Error term vector, for the difference in each y-hat and y
    errors = (X @ theta) - y

    #Squaring and summing
    errors_sqred = errors.T @ errors
    
    #Average
    mean_errors_sqrd = errors_sqred / (X.shape[0])

    return mean_errors_sqrd[0, 0]

def gradient_descent(X:np.ndarray, y:np.ndarray, theta:np.ndarray, alpha:float, epochs):
    cost_array = [mse_cost(X, y, theta)]

    num_examples = X.shape[0]
    gradient = np.zeros((1,2))
    for iteration in range(epochs):
        
        # for each i
        error = (X @ theta) - y
        gradient = X.T @ error
        gradient = gradient / num_examples
        theta -= alpha * gradient
        
        iteration_cost = mse_cost(X, y, theta)
        cost_array.append(iteration_cost)

        print(f"Iteration {iteration+1} cost: {iteration_cost}")

    plot_cost(range(epochs+1), cost_array)
    

def plot_cost(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.show()




    

    


# PREPARATION
coordinates = np.array([[0,0], [1,-1]], dtype=float)
X, y = prepare_data(coordinates)
# Appending the column of 1s; will correspond with the first theta bias term when multiplied
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

# Initial stats
random_slope = np.random.randint(0, 6)
theta = np.array([[0], [1]], dtype=float)
print(f"Initial parameters: {theta}")
initial_cost = mse_cost(X, y, theta)
print(f"Initial cost: {initial_cost}")

gradient_descent(X, y, theta, 0.03, 1000)
print()

# Final stats

print(f"Final parameters: {theta}")
final_cost = mse_cost(X, y, theta)
print(f"Final cost: {final_cost}")
print(f"Final Line: {theta[1][0]}x + {theta[0][0]}")

test = "asdf"
print(test[:(len(test) - 1)])