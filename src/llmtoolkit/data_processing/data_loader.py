def load_data_from_txt(file_path: str) -> List[str]:
    """
    Load data from a .txt file.
    
    Args:
        file_path (str): Path to the data file.
    
    Returns:
        List[str]: Loaded data lines.
    """
    with open(file_path, 'r') as file:
        return file.readlines()
