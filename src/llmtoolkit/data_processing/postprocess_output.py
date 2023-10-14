from typing import List, Union
from transformers import PreTrainedTokenizer, BertTokenizer, GPT2Tokenizer, T5Tokenizer, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer

MODEL_TOKENIZERS = {
    "bert": BertTokenizer,
    "gpt2": GPT2Tokenizer,
    "t5": T5Tokenizer,
    "roberta": RobertaTokenizer,
    "xlnet": XLNetTokenizer,
    "albert": AlbertTokenizer,
    # ... Add more as needed
}

def remove_special_tokens(tokens: List[str], model_type: str, model_name: str) -> List[str]:
    """
    Remove special tokens associated with a given pre-trained model.

    Args:
    - tokens (List[str]): List of tokens to be cleaned.
    - model_type (str): Type of pre-trained model (e.g., "bert", "gpt2").
    - model_name (str): Name or path of the pre-trained model (e.g., "bert-base-uncased").

    Returns:
    - List[str]: Cleaned list of tokens.
    """

    if model_type not in MODEL_TOKENIZERS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {list(MODEL_TOKENIZERS.keys())}")

    tokenizer = MODEL_TOKENIZERS[model_type].from_pretrained(model_name)
    
    special_tokens = tokenizer.all_special_tokens
    return [token for token in tokens if token not in special_tokens]

def detokenize(token_ids: List[int], model_type: str, model_name: str, skip_special_tokens: bool = True) -> str:
    """
    Convert a list of token IDs back to a string using the tokenizer's decode method.

    Args:
    - token_ids (List[int]): List of token IDs to be decoded.
    - model_type (str): Type of pre-trained model (e.g., "bert", "gpt2").
    - model_name (str): Name or path of the pre-trained model (e.g., "bert-base-uncased").
    - skip_special_tokens (bool): Whether to remove special tokens like [PAD], [CLS], etc.

    Returns:
    - str: Decoded string.
    """

    if model_type not in MODEL_TOKENIZERS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {list(MODEL_TOKENIZERS.keys())}")

    tokenizer = MODEL_TOKENIZERS[model_type].from_pretrained(model_name)
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

# Further functions for postprocessing can be added here.
