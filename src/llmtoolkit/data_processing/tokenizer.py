def tokenize_data(data: List[str], tokenizer):
    """
    Tokenize data using a given tokenizer.
    
    Args:
        data (List[str]): List of data samples.
        tokenizer: Tokenizer instance (e.g., from HuggingFace's transformers library).
    
    Returns:
        List: Tokenized data.
    """
    return [tokenizer.encode(sample) for sample in data]
