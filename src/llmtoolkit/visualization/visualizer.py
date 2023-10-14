import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_loss_curve(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    
    Args:
    - train_losses (list): Training losses over epochs.
    - val_losses (list): Validation losses over epochs.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, '-o', label='Training Loss')
    plt.plot(epochs, val_losses, '-o', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_word_distribution(texts, top_n=30):
    """
    Plot word distribution for the top N words in the texts.
    
    Args:
    - texts (list of str): Text data.
    - top_n (int): Number of top words to display.
    """
    words = [word for text in texts for word in text.split()]
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[count for word, count in most_common_words], 
                y=[word for word, count in most_common_words])
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.title(f'Top {top_n} Most Common Words')
    plt.grid(True, axis='x')
    plt.show()

def compare_predictions(predictions, ground_truth, samples=5):
    """
    Display model's predictions against the ground truth for a few samples.
    
    Args:
    - predictions (list of str): Model's predictions.
    - ground_truth (list of str): Actual values.
    - samples (int): Number of samples to display.
    """
    for i in range(samples):
        print(f"Sample {i + 1}:")
        print("Prediction:", predictions[i])
        print("Ground Truth:", ground_truth[i])
        print("-" * 50)
