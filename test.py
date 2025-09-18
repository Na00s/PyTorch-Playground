import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def speculative_decode(target_model, draft_model, prompt_idx, max_new_tokens, gamma, eps=1e-12):
    target_model.eval()
    draft_model.eval()

    B, T = prompt_idx.shape
    idx = prompt_idx.clone()

    context_length = min(draft_model.config.context_length, target_model.config.context_length)

    while idx.shape[1] < T + max_new_tokens:

        current_context = idx[:, -context_length:]

        #speculation
        draft_tokens_list = []
        draft_tokens_probs = []
        draft_context = current_context
        for _ in range(gamma):
            draft_tokens_logits, _ = draft_model(draft_context)
            next_token_logits = draft_tokens_logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            next_token_prob = torch.gather(next_token_probs, -1, next_token)
            draft_tokens_list.append(next_token)
            draft_tokens_probs.append(next_token_prob)
            draft_context = torch.cat([draft_context, next_token], dim=-1)
        draft_tokens = torch.cat(draft_tokens_list, dim=-1)
        p_draft = torch.cat(draft_tokens_probs, dim=-1)

        #verification
        full_targets_logits, _, = target_model(draft_context)
        start_pos = current_context.shape[1]
        targets_logits = full_targets_logits[:, start_pos-1:-1, :]
        targets_probs = F.softmax(targets_logits, dim=-1)
        p_target = torch.gather(targets_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)

        alpha = (p_target / (p_draft + eps)).clamp(max=1.0)
        u = torch.rand_like(alpha)

        accepted_mask = (u <= alpha)
        prefix_mask = accepted_mask.int().cumprod(dim=-1).bool()
        acceptance_ratio = prefix_mask.sum(dim=-1) / gamma
        print(acceptance_ratio)
        min_n_accepted = min(prefix_mask.sum(dim=-1)).item()

        if min_n_accepted > 0:
            idx = torch.cat([idx, draft_tokens[:, :min_n_accepted]])
        if idx.shape[1] >= T + max_new_tokens:
            break
        if min_n_accepted < gamma:
            corrected_probs = targets_probs[:, min_n_accepted, :]
            corrected_token = torch.multinomial(corrected_probs, num_samples=1)
            idx = torch.cat([idx, corrected_token], dim=-1)
        else:
            bonus_logits = full_targets_logits[:, -1, :]
            bonus_probs = F.softmax(bonus_logits, dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            idx = torch.cat([idx, bonus_token], dim=-1)
    
    return idx[:, :T+max_new_tokens]

class KLDivLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction="batchmean"):
        super().__init__()
        self.T = temperature
        self.r = reduction
    def forward(self, student_logits, teacher_logits):
        student_logits = student_logits / self.T
        teacher_logits = teacher_logits / self.T
        
        log_softmax_student = F.log_softmax(student_logits, dim=-1)
        log_softmax_teacher = F.log_softmax(teacher_logits, dim=-1)

        loss = F.kl_div(log_softmax_student, log_softmax_teacher, reduction="none", log_target=True)

        loss = loss.sum(dim=-1)

        if self.r == "mean":
            loss = loss.mean()
        elif self.r == "batchmean":
            loss = loss.sum() / teacher_logits.shape[0]
        elif self.r == "sum":
            loss = loss.sum()

        return loss

from torchtune.models.llama3 import llama3
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

# Smaller version of llama3 to fit in a single GPU
model = llama3(
    vocab_size=4096,
    num_layers=16,
    num_heads=16,
    num_kv_heads=4,
    embed_dim=2048,
    max_seq_len=2048,
).cuda()

# Quantizer for int8 dynamic per token activations +
# int4 grouped per channel weights, only for linear layers
qat_quantizer = Int8DynActInt4WeightQATQuantizer()

# Insert "fake quantize" operations into linear layers.
# These operations simulate quantization numerics during
# training without performing any dtype casting
model = qat_quantizer.prepare(model)

# Standard training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(10):
    example = torch.randint(0, 4096, (2, 16)).cuda()
    target = torch.randn((2, 16, 4096)).cuda()
    output = model(example)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Convert fake quantize to actual quantize operations
# The quantized model has the exact same structure as the
# quantized model produced in the corresponding PTQ flow
# through `Int8DynActInt4WeightQuantizer`
model = qat_quantizer.convert(model)







        
        













