#A collection of useful helper functions for linear algebra operations.

import numpy as np
import scipy.linalg as la

def block_toeplitz(blocks):
    """
    Create a block Toeplitz matrix from a list of blocks.
    
    Parameters:
    blocks (list of np.ndarray): List of 2D arrays representing the blocks.
    
    Returns:
    np.ndarray: Block Toeplitz matrix.
    """
    # Determine the size of each block and the number of blocks
    block_shape = blocks[0].shape
    num_blocks = len(blocks)
    
    # Initialize the block Toeplitz matrix
    toeplitz_matrix = np.zeros((block_shape[0] * num_blocks, block_shape[1] * num_blocks))
    
    # Fill in the block Toeplitz matrix
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i >= j:
                toeplitz_matrix[i*block_shape[0]:(i+1)*block_shape[0], j*block_shape[1]:(j+1)*block_shape[1]] = blocks[i-j].T
            else:
                toeplitz_matrix[i*block_shape[0]:(i+1)*block_shape[0], j*block_shape[1]:(j+1)*block_shape[1]] = blocks[j-i]
    
    return toeplitz_matrix

def print_matrix(matrix, name):
    """
    Print a matrix in a readable format.
    
    Parameters:
    matrix (np.ndarray): The matrix to print.
    name (str): The name of the matrix.
    """
    print(f"Matrix {name}:")
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            print(f"{matrix[i, j]:>10.4f}", end=" ")
        print()

def create_repeated_block_diagonal(mat, num_blocks):
    """
    Create a block diagonal matrix with matrix mat repeated along the diagonal and zeros elsewhere.
    
    Parameters:
    mat (np.ndarray): The matrix to place along the diagonal.
    num_blocks (int): The number of times to place mat along the diagonal.
    
    Returns:
    np.ndarray: The block diagonal matrix.
    """
    # Determine the shape of the block matrix
    block_shape = mat.shape
    block_size = block_shape[0] * num_blocks
    
    # Initialize the block diagonal matrix with zeros
    block_diagonal_matrix = np.zeros((block_size, block_size))
    
    # Fill in the block diagonal matrix with mat along the diagonal
    for i in range(num_blocks):
        start_index = i * block_shape[0]
        end_index = start_index + block_shape[0]
        block_diagonal_matrix[start_index:end_index, start_index:end_index] = mat
    
    return block_diagonal_matrix

def lower_threshold_matrix(matrix, threshold):
    """
    Set all entries below the threshold to zero in a matrix.
    
    Parameters:
    matrix (np.ndarray): The matrix to threshold.
    threshold (float): The threshold value.
    
    Returns:
    np.ndarray: The thresholded matrix.
    """
    thresholded_matrix = matrix.copy()
    thresholded_matrix[np.abs(thresholded_matrix) < threshold] = 0
    return thresholded_matrix

def compute_powers(matrix, n):
    # Initialize the array to hold the powers of the matrix
    powers = np.empty((n, *matrix.shape), dtype=matrix.dtype)
    
    # Compute each power of the matrix
    current_power = np.eye(matrix.shape[0], dtype=matrix.dtype)  # Start with the identity matrix
    for i in range(n):
        powers[i] = current_power
        current_power = current_power@matrix
    
    return powers