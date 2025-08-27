import torch
from model import device, LLAMA3_CONFIG, TransformerModel

with open("big.txt", "r", encoding="utf-8") as f:
    data = f.read()

vocab = sorted(list(set(data)))
vocab_size = len(vocab)

block_size = 512
config = LLAMA3_CONFIG(max_position_embeddings=block_size, vocab_size=vocab_size)
m = TransformerModel(config).to(device=device, dtype=config.torch_dtype)

batch_size = 8
num_steps = 500
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}

encode = lambda s: [stoi[char] for char in s]
decode = lambda n: "".join([itos[i] for i in n])

data = torch.tensor(encode(data), dtype=torch.long)

training_cutoff = int(0.9*len(data))

data_train = data[ : training_cutoff]
data_val = data[training_cutoff : ]

def get_batch(split):
    data = data_train if split == "train" else data_val
    indices = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1: i + 1+ block_size] for i in indices])
    return x.to(device), y.to(device)

for i in range(num_steps):
    optimizer.zero_grad(set_to_none=True)
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    print(loss)
    loss.backward()
    optimizer.step()


