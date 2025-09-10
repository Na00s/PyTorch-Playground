import torch
import math 

eps = 1e-12

# B = 1 (unbatched) (T=2, V=4) - CONSISTENT VERSION
p_draft = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.25, 0.5, 0.15, 0.1]])
p_target = torch.tensor([[0.08, 0.22, 0.3, 0.4],
                         [0.2, 0.55, 0.15, 0.1]])

T, V = p_draft.shape

y = torch.multinomial(p_draft, num_samples=1).squeeze(1) #T

p_draft_chosen = p_draft[torch.arange(T), y] #T
p_target_chosen = p_target[torch.arange(T), y] #T

u = torch.rand(T) #T

alpha = (p_target_chosen / (p_draft_chosen + eps)).clamp(max=1.0) #T

accepted_mask = (u <= alpha) #T

prefix_mask = accepted_mask.int().cumprod(dim=0).bool() #T

number_of_committed_tokens = prefix_mask.sum().item() #scalar

# Process similar to batched version
l = number_of_committed_tokens
if l == T:
    committed_tokens = y
    advance = l
else:
    z = torch.multinomial(p_target[l], num_samples=1).squeeze(0)
    committed_tokens = torch.cat([y[:l], z.view(1)])
    advance = l + 1

print("Unbatched results:")
print(f"Sampled tokens: {y}")
print(f"Committed tokens: {committed_tokens}")
print(f"Advance: {advance}")
print(f"Acceptance mask: {accepted_mask}")
print(f"Prefix mask: {prefix_mask}")
print()

