import torch
import math 

eps = 1e-12

# B = 3 (batched), (T=4, V=4) - FOR COMPARISON
p_draft = torch.tensor([
    [[0.1, 0.2, 0.3, 0.4],
     [0.25, 0.5, 0.15, 0.1],
     [0.4, 0.4, 0.1, 0.1],
     [0.3, 0.3, 0.2, 0.2]],

    [[0.05, 0.25, 0.25, 0.45],
     [0.1, 0.6, 0.2, 0.1],
     [0.3, 0.3, 0.2, 0.2],
     [0.5, 0.2, 0.2, 0.1]],

    [[0.4, 0.3, 0.2, 0.1],
     [0.2, 0.2, 0.4, 0.2],
     [0.1, 0.2, 0.3, 0.4],
     [0.25, 0.25, 0.25, 0.25]]
], dtype=torch.float32)

p_target = torch.tensor([
    [[0.08, 0.22, 0.3, 0.4],
     [0.2, 0.55, 0.15, 0.1],
     [0.35, 0.35, 0.15, 0.15],
     [0.25, 0.25, 0.25, 0.25]],

    [[0.1, 0.2, 0.3, 0.4],
     [0.15, 0.55, 0.2, 0.1],
     [0.25, 0.25, 0.25, 0.25],
     [0.4, 0.3, 0.2, 0.1]],

    [[0.35, 0.25, 0.25, 0.15],
     [0.25, 0.25, 0.25, 0.25],
     [0.15, 0.25, 0.25, 0.35],
     [0.1, 0.2, 0.3, 0.4]]
], dtype=torch.float32)

B, T, V = p_draft.shape

p_draft_flat = p_draft.reshape(B*T, V) #B*T, V
p_target_flat = p_target.reshape(B*T, V) #B*T, V

y_flat = torch.multinomial(p_draft_flat, num_samples=1).squeeze(1) #B*T
y = y_flat.reshape(B, T) #B, T

batch_idx = torch.arange(B).unsqueeze(1).expand(B, T) #B, T
time_idx = torch.arange(T).unsqueeze(0).expand(B, T) #B, T

p_draft_chosen = p_draft[batch_idx, time_idx, y] #B, T
p_target_chosen = p_target[batch_idx, time_idx, y] #B, T

alpha = (p_target_chosen / (p_draft_chosen + eps)).clamp(max=1.0) #B, T

u = torch.rand(B, T) #B, T

accepted_mask = (u <= alpha) #B, T

prefix_mask = accepted_mask.int().cumprod(dim=1).bool() #B, T

number_of_committed_tokens = prefix_mask.sum(dim=1) #B

committed = []
advance = []

for b in range(B):
    l = number_of_committed_tokens[b].item()
    if l == T:
        committed.append(y[b])
        advance.append(l)
    else:
        z = torch.multinomial(p_target[b, l], num_samples=1).squeeze(0)
        committed.append(torch.cat([y[b, :l], z.view(1)]))
        advance.append(l+1)

print("Batched results:")
for b in range(B):
    print(f"Batch {b}:")
    print(f"  Sampled tokens: {y[b]}")
    print(f"  Committed tokens: {committed[b]}")
    print(f"  Advance: {advance[b]}")
    print(f"  Acceptance mask: {accepted_mask[b]}")
    print(f"  Prefix mask: {prefix_mask[b]}")
    print()

# Optional: Summary statistics
print("Summary:")
print(f"Total batches: {B}")
print(f"Average advance: {sum(advance) / B:.2f}")
print(f"Advance counts: {advance}")
print(f"Number of committed tokens per batch: {number_of_committed_tokens.tolist()}")

# Optional: Show acceptance rates
acceptance_rates = accepted_mask.float().mean(dim=1)
print(f"Acceptance rates per batch: {acceptance_rates.tolist()}")






















                                   

