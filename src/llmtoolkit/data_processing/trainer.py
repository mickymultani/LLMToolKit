import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AdamW, get_linear_schedule_with_warmup

from llmtoolkit.data_processing.data_loader import load_data_from_txt
from llmtoolkit.data_processing.tokenizer import tokenize_texts
from llmtoolkit.data_processing.data_splitter import split_data

# Define paths
DATA_PATH = './LLMToolKit/src/llmtoolkit/data/raw/input_data.txt'
SAVE_DIR = './LLMToolKit/src/llmtoolkit/models/'

# 1. Load Data:
texts = load_data_from_txt(DATA_PATH)

# 2. Tokenize Data:
model_type = "gpt2"  # or "bert", "t5", etc.
model_name = "gpt2-medium"  # adjust to the specific pretrained model you are using
tokenized_texts = tokenize_texts(texts, model_type, model_name)

# 3. Convert Tokenized Data to TensorDataset:
input_ids = [torch.tensor(token_ids) for token_ids in tokenized_texts]
dataset = TensorDataset(torch.stack(input_ids))

# 4. Split Dataset:
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 5. Create DataLoader:
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 6. Define Model, Optimizer, and Scheduler:
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(model_name)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)  # 3 epochs

# 7. Training Loop:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # 3 epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch[0].to(device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs, labels=inputs)
            val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1} validation loss: {val_loss}")

# 8. Save Model:
model.save_pretrained(SAVE_DIR)