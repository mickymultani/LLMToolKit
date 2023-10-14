from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer
from typing import List, Union, Optional, Tuple
from transformers import AddedToken


MODEL_TOKENIZERS = {
    "bert": BertTokenizer,
    "gpt2": GPT2Tokenizer,
    "t5": T5Tokenizer,
    "roberta": RobertaTokenizer,
    "xlnet": XLNetTokenizer,
    "albert": AlbertTokenizer,
    # ... Add more as needed
}

def tokenize_texts(texts: List[str], model_type: str, model_name: str, 
                   model_max_length: Optional[int] = None, 
                   padding_side: Optional[str] = None,
                   truncation_side: Optional[str] = None,
                   chat_template: Optional[str] = None,
                   model_input_names: Optional[List[str]] = None,
                   bos_token: Optional[Union[str, AddedToken]] = None,
                   eos_token: Optional[Union[str, AddedToken]] = None,
                   unk_token: Optional[Union[str, AddedToken]] = None,
                   sep_token: Optional[Union[str, AddedToken]] = None,
                   pad_token: Optional[Union[str, AddedToken]] = None,
                   cls_token: Optional[Union[str, AddedToken]] = None,
                   mask_token: Optional[Union[str, AddedToken]] = None,
                   additional_special_tokens: Optional[Tuple[Union[str, AddedToken], ...]] = None,
                   clean_up_tokenization_spaces: bool = True,
                   split_special_tokens: bool = False
                  ) -> List[List[str]]:
    """
    Extended documentation...
    """
    
    if model_type not in MODEL_TOKENIZERS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported models are: {list(MODEL_TOKENIZERS.keys())}")

    tokenizer = MODEL_TOKENIZERS[model_type].from_pretrained(
        model_name,
        model_max_length=model_max_length,
        padding_side=padding_side,
        truncation_side=truncation_side,
        chat_template=chat_template,
        model_input_names=model_input_names,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unk_token,
        sep_token=sep_token,
        pad_token=pad_token,
        cls_token=cls_token,
        mask_token=mask_token,
        additional_special_tokens=additional_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        split_special_tokens=split_special_tokens
    )
    
    return [tokenizer.tokenize(text) for text in texts]
