import torch
import math

def get_sliding_windows(image, window_size=1024):
    """
    Extract overlapping sliding windows from an image tensor.
    
    Args:
        image: torch.Tensor of shape (channels, height, width)
        window_size: int, size of square windows (default 1024)
    
    Returns:
        windows: list of torch.Tensor, each of shape (channels, window_size, window_size)
        positions: list of tuples (y, x) indicating top-left corner of each window
        grid_shape: tuple (num_windows_h, num_windows_w)
        strides: tuple (stride_h, stride_w)
    """
    channels, height, width = image.shape
    
    # Calculate number of windows needed in each dimension
    num_windows_h = math.ceil(height / window_size)
    num_windows_w = math.ceil(width / window_size)
    
    # Calculate stride for each dimension
    if num_windows_h > 1:
        stride_h = (height - window_size) / (num_windows_h - 1)
    else:
        stride_h = 0
    
    if num_windows_w > 1:
        stride_w = (width - window_size) / (num_windows_w - 1)
    else:
        stride_w = 0
    
    windows = []
    positions = []
    
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            # Calculate top-left position
            y = int(round(i * stride_h))
            x = int(round(j * stride_w))
            
            # Ensure we don't go out of bounds (due to rounding)
            y = max(0, min(y, height - window_size))
            x = max(0, min(x, width - window_size))
            
            # Extract window
            window = image[:, y:y+window_size, x:x+window_size]
            
            windows.append(window)
            positions.append((y, x))
    
    return windows, positions

def reconstruct_from_windows(windows, positions, original_shape, window_size=1024):
    """
    Reconstruct the original image from overlapping windows using weighted averaging.
    
    Args:
        windows: list of torch.Tensor, each of shape (channels, window_size, window_size)
        positions: list of tuples (y, x) indicating top-left corner of each window
        original_shape: tuple (channels, height, width) of the original image
        window_size: int, size of square windows (default 1024)
    
    Returns:
        reconstructed: torch.Tensor of shape (channels, height, width)
    """
    height, width = original_shape
    
    # Initialize output tensor
    reconstructed = torch.zeros(original_shape, dtype=windows[0].dtype, device=windows[0].device)
    
    # Add each window to the reconstruction
    for window, (y, x) in zip(windows, positions):
        window_h, window_w = window.shape[-2:]
        y_start = max(0, y)
        x_start = max(0, x)
        y_end = min(y_start + window_h, height)
        x_end = min(x_start + window_w, width)

        # Crop window if needed to match destination slice
        cropped = window[: y_end - y_start, : x_end - x_start]
        reconstructed[y_start:y_end, x_start:x_end] = cropped
        
    return reconstructed
