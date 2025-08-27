import os
import math
import time
import torch
import torch.nn.functional as F

from cutecharts.charts import Line
from model import GEMMA3_270M_CONFIG, TransformerModel, device

torch.manual_seed(1337)

batch_size = 8
num_steps = 500
learning_rate = 1e-3
block_size = 512

# data read
with open("big.txt", "r", encoding="utf-8") as f:
    data = f.read()

# character level vocab
vocab = sorted(list(set(data)))
data_vocab_size = len(vocab)

config = GEMMA3_270M_CONFIG()
config.vocab_size = data_vocab_size

m = TransformerModel(config, config.layer_type).to(device=device, dtype=config.dtype)
m.proj_o.weight = m.embedder.embed_table.weight 

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Tokenization
itos = {i : ch for i, ch in enumerate(vocab)}
stoi = {ch : i for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[char] for char in s]
decode = lambda nums: "".join([itos[i] for i in nums])

# Split
data_tensor = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9*len(data_tensor))

train_data = data_tensor[ : n]
val_data = data_tensor[n : ]

# Batch preparation
def get_batch(split):
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:block_size+i] for i in indices])
    y = torch.stack([data[i+1:block_size+i+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y

train_losses, val_losses = [], []

for _ in range(num_steps):
    x_train, y_train = get_batch("train")
    logits, train_loss = m(x_train, y_train)
    print(train_loss.item())
    train_losses.append(train_loss.item())
    optimizer.zero_grad(set_to_none=True)
    train_loss.backward()
    if num_steps % 10 == 0:
        m.eval()
        x_val, y_val = get_batch("val")
        _, val_loss = m(x_val, y_val)
        val_losses.append(val_loss.item())
        m.train()
    optimizer.step()









