import os
import math
import torch

from vocab_aligner import DraftModelVocabAligned
from model import TransformerModel, GEMMA3_270M_CONFIG, device

import torch.nn as nn
import torch.nn.functional as F

with open("big.txt", "r", encoding="utf-8") as f:
    data = f.read()

vocab = sorted(list(set(data)))

itos = {i:s for i, s in enumerate(vocab)}
stoi = {s:i for i, s in enumerate(vocab)}

encode = lambda s: [stoi[char] for char in s]
decode = lambda i: "".join([itos[num] for num in i])

training_cutoff = int(0.9*len(data))

train_data = torch.tensor(encode(data[:training_cutoff]), dtype=torch.long)
val_data = torch.tensor(encode(data[training_cutoff:]), dtype=torch.long)

TargetConfig = GEMMA3_270M_CONFIG(vocab_size=len(vocab))
DraftConfig = GEMMA3_270M_CONFIG(vocab_size = len(vocab), n_layers=9, layer_types=TargetConfig.layer_types[:9])
TargetModel = TransformerModel(TargetConfig).to(device)
DraftModel = TransformerModel(DraftConfig).to(device)

TargetModel_num_params = sum([p.numel() for p in TargetModel.parameters()])
DraftModel_num_params = sum([p.numel() for p in DraftModel.parameters()])

print(device, TargetModel_num_params, DraftModel_num_params, ", No vocab projection weight tying")

DraftModelAligned = DraftModelVocabAligned(TargetModel, DraftModel, freeze_draft=False).to(device)

optimizer = torch.optim.AdamW(DraftModelAligned.parameters(), lr=1e-3)
num_steps = 500
batch_size = 8
block_size = 100

def get_batch(split):
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    x = x.to(device)
    y = y.to(device)
    return x, y

class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean", eps: float = 1e-12):
        super().__init__()
        assert reduction in ("none", "sum", "mean", "batchmean")
        self.T = float(temperature)
        self.eps = float(eps)
        self.reduction = reduction
    def forward(self, teacher_logits, student_logits):
        assert teacher_logits.shape == student_logits.shape
        B, T, V = teacher_logits.shape
        teacher_probs = F.softmax(teacher_logits / self.T, dim=-1) #B, T, V
        student_probs = torch.clamp(F.softmax(student_logits / self.T, dim=-1), min=self.eps) #B, T, V
        loss = teacher_probs * torch.log(teacher_probs / student_probs) #B, T, V
        loss = loss.sum(dim=-1) #B, T

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "batchmean":
            loss = loss.sum() / teacher_logits.shape[0]
        return loss

loss = KLDivergenceLoss()

for p in TargetModel.parameters():
    p.requires_grad = False

KL_losses = []
TargetModel.eval()
DraftModelAligned.train()
for i in range(num_steps):
    optimizer.zero_grad(set_to_none=True)
    train_data_x, train_data_y = get_batch("train")
    with torch.no_grad():
        teacher_logits, _ = TargetModel(train_data_x)
    student_logits, _ = DraftModelAligned(train_data_x)
    kl_loss = loss(teacher_logits, student_logits)
    print(kl_loss.item())
    KL_losses.append(kl_loss.item())
    kl_loss.backward()
    optimizer.step()
        


torch.save(DraftModelAligned.state_dict(), "draft_model_aligned_distilled.pth")
    
        
