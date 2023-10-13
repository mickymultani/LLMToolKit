from typing import Tuple, List

def split_data(data: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data (List[str]): List of data samples.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
    
    Returns:
        Tuple[List[str], List[str], List[str]]: Training, validation, and test sets.
    """
    total_samples = len(data)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
