import numpy as np

def compute_mse(original, predicted):
    """
    Compute Mean Squared Error (MSE) between original and predicted frame
    
    :param original: original frame
    :param predicted: predicted frame
    :return: MSE value
    """
    # Convert to int32 to prevent overflow
    diff = original.astype(np.int32) - predicted.astype(np.int32)
    
    # Calculate mean of squared differences
    mse = np.mean(diff ** 2)
    
    return mse