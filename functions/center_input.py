import numpy as np

def center_input(input):
    # Get the coordinates of all non-white elements
    non_zero_coords = np.argwhere(input != 255)
    
    if non_zero_coords.size == 0:
        return np.zeros_like(input)  # Return a blank array if no non-white elements exist
    
    # Get the bounds of the non-white area
    min_row, min_col = non_zero_coords.min(axis=0)
    max_row, max_col = non_zero_coords.max(axis=0)
    
    # Determine the side length of the smallest square that can encompass all non-white elements
    side_length = max(max_row - min_row, max_col - min_col) + 1
    
    # Calculate the square subarray bounds
    start_row = (min_row + max_row) // 2 - side_length // 2
    start_col = (min_col + max_col) // 2 - side_length // 2
    end_row = start_row + side_length
    end_col = start_col + side_length

    # Extract the cropped square, ensuring it does not exceed the original array dimensions
    cropped = input[max(0, start_row):min(end_row, input.shape[0]), 
                    max(0, start_col):min(end_col, input.shape[1])]

    # Create a blank square array with the smallest required dimensions
    centered = np.full((side_length, side_length), 255, dtype=input.dtype)

    # Calculate offsets to place the cropped image in the center of the blank square
    offset_row = (side_length - cropped.shape[0]) // 2
    offset_col = (side_length - cropped.shape[1]) // 2

    # Place the cropped square in the center of the new square
    centered[offset_row:offset_row + cropped.shape[0], 
             offset_col:offset_col + cropped.shape[1]] = cropped
    
    return centered
