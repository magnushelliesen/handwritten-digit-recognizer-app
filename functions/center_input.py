import numpy as np

def center_input(input):
    # Get the coordinates of all non-white elements
    non_zero_coords = np.argwhere(input != 255)
    
    if non_zero_coords.size == 0:
        return np.array([])
    
    # Get the bounds of the non-white area
    min_row, min_col = non_zero_coords.min(axis=0)
    max_row, max_col = non_zero_coords.max(axis=0)
    
    # Determine the side length of the smallest square that can encompass all non-white elements
    side_length = max(max_row-min_row, max_col-min_col)+1
    
    # Calculate the square subarray bounds
    start_row, start_col = min_row, min_col
    end_row, end_col = start_row+side_length, start_col+side_length
    
    # Handle cases where the bounds may exceed the original array dimensions
    end_row = min(end_row, input.shape[0])
    end_col = min(end_col, input.shape[1])
    
    return input[start_row:end_row, start_col:end_col]