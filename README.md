# LLMToolkit

A comprehensive toolkit designed for seamless integration and utilization of Large Language Models (LLMs). 

## Features

- **Data Processing**: Simplified data preparation and transformation for LLMs.
- **Evaluation Metrics**: Evaluate the performance of your models using diverse metrics.
- **Visualizers**: Obtain insights and visualizations related to your model's training and predictions.
- **Utilities**: Handy utilities to enhance your workflow with popular platforms.
- ...and more to come!

## Installation

To install the `LLMToolkit`, use pip:

```python
pip install llmtoolkit
```

## Quickstart

### 1. Data Preparation

```python
from llmtoolkit.data_processing import load_data_from_txt, tokenize_texts

# Load your data
texts = load_data_from_txt("path_to_your_text_data.txt")

# Tokenize your data
tokenized_texts = tokenize_texts(texts, model_type="gpt2", model_name="gpt2-medium")
```

### 2. Training
Here's a simplified training example for fine-tuning a GPT-2 model:

```python
from llmtoolkit.train import train_model

train_model(data_path="path_to_your_text_data.txt", model_type="gpt2", model_name="gpt2-medium", epochs=3)
```

### 3. Evaluations
Easily evaluate your model's performance:

```python
from llmtoolkit.evaluations import compute_bleu, compute_rouge

reference = ["This is a sample reference text."]
hypothesis = ["This is a sample hypothesis text."]

bleu_score = compute_bleu(reference, hypothesis)
rouge_score = compute_rouge(reference[0], hypothesis[0])
```

### 4. Visualizations
Visualize your model's training progress and more:

```python
from llmtoolkit.visualizer import plot_loss_curve

plot_loss_curve(training_losses, validation_losses)
```

## Documentation
More detailed documentation on each module, function, and its usage is available in the docs directory.

## Support
Having issues or suggestions? Open a new issue here or contact us directly!

## License
This project is licensed under the MIT License.