import torch
import torch.nn as nn
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_TOKENIZERS = {
    "bert": BertTokenizer,
    "gpt2": GPT2Tokenizer,
    "t5": T5Tokenizer,
    "roberta": RobertaTokenizer,
    "xlnet": XLNetTokenizer,
    "albert": AlbertTokenizer
}

# Utility function for tokenization
def tokenize_text(text, model_type="gpt2"):
    tokenizer = MODEL_TOKENIZERS[model_type].from_pretrained(model_type)
    return tokenizer.tokenize(text)

# Initialize the cross-entropy loss function
cross_entropy_loss = nn.CrossEntropyLoss()

def compute_perplexity(logits, targets):
    """
    Computes the perplexity based on the logits and targets.
    
    Args:
    - logits (torch.Tensor): Model predictions.
    - targets (torch.Tensor): True labels.
    
    Returns:
    - float: The computed perplexity.
    """
    
    # Check shapes and try to reshape if needed
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(1)  # Add a singleton dimension for num_classes
    elif len(logits.shape) == 3:
        if logits.shape[1] < logits.shape[2]:  # If sequence_length > num_classes
            logits = logits.transpose(1, 2)  # Swap sequence_length and num_classes
    else:
        raise ValueError("Logits tensor should be 2D or 3D.")
    
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)
    elif len(targets.shape) > 2:
        raise ValueError("Targets tensor should be 1D or 2D.")
    
    loss = cross_entropy_loss(logits, targets)
    return torch.exp(loss).item()


def compute_bleu(reference, hypothesis, ngram=4):
    weights = [1.0/ngram]*ngram
    return sentence_bleu([reference], hypothesis, weights=weights)

def compute_corpus_bleu(references, hypotheses, ngram=4):
    weights = [1.0/ngram]*ngram
    return corpus_bleu(references, hypotheses, weights=weights)

def compute_rouge(reference, hypothesis):
    rouge = Rouge()
    return rouge.get_scores(hypothesis, reference, avg=True)

def compute_f1(reference_labels, hypothesis_labels):
    return f1_score(reference_labels, hypothesis_labels, average='weighted')

def compute_precision(reference_labels, hypothesis_labels):
    return precision_score(reference_labels, hypothesis_labels, average='weighted')

def compute_recall(reference_labels, hypothesis_labels):
    return recall_score(reference_labels, hypothesis_labels, average='weighted')

def compute_word_overlap(reference, hypothesis):
    ref_set, hypo_set = set(reference), set(hypothesis)
    overlap = ref_set.intersection(hypo_set)
    
    ref_overlap = len(overlap) / len(ref_set) if ref_set else 0.0
    hypo_overlap = len(overlap) / len(hypo_set) if hypo_set else 0.0
    
    return {"Ref Overlap": ref_overlap, "Hypo Overlap": hypo_overlap}
