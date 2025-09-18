import math
import torch
import torch.nn as nn
import torch.nn.functional as F

with open("big.txt", "r", encoding="utf-8") as f:
    data = f.read()

vocab = sorted(list(set(data)))
vocab_size = len(vocab)

itos = {i: s for i, s in enumerate(vocab)}
stoi = {s: i for i, s in enumerate(vocab)}

encode = lambda s: [stoi[char] for char in s]
decode = lambda num: "".join([itos[i] for i in num])

def speculative_decode(target_model, draft_model, prompt_idx, max_new_tokens, gamma):
    target_model.eval()
    draft_model.eval()

    B, T = prompt_idx.shape
    idx = prompt_idx.clone()
    context_length = min(target_model.config.context_length, draft_model.config.context_length)

    while idx.shape[1] < T + max_new_tokens:

        # defining shared context
        current_context = idx[:, -context_length:]

        # drafting
        draft_context = current_context
        p_draft_list = []
        draft_tokens_list = []
        for _ in range(gamma):
            draft_logits, _ = draft_model(draft_context)
            next_token_logits = draft_logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            next_token_prob = torch.gather(next_token_probs, -1, next_token)
            p_draft_list.append(next_token_prob)
            draft_tokens_list.append(next_token)
            draft_context = torch.cat([draft_context, next_token], dim=-1)
        p_draft = torch.cat(p_draft_list, dim=-1)
        draft_tokens = torch.cat(draft_tokens_list, dim=-1)

        # verifying
        full_target_logits, _ = target_model(draft_context) #(B, initial context + gamma new tokens + bonus, V), _
        start_pos = current_context.shape[1]
        target_logits = full_target_logits[:, start_pos-1:-1, :]
        target_probs = F.softmax(target_logits, dim=-1)
        p_target = torch.gather(target_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)

        alpha = (p_target / p_draft + 1e-12).clamp(max=1.0)
        u = torch.rand_like(alpha)

        accepted_mask = ( u <= alpha)
        prefix_mask = accepted_mask.int().cumprod(dim=1).bool()
        number_of_accepted_tokens = prefix_mask.sum(dim=1)
        acceptance_ratio = number_of_accepted_tokens / (B*gamma)

        # updating the sequence
        min_n_accepted = number_of_accepted_tokens.min().item()

        if min_n_accepted > 0:
            idx = torch.cat([idx, draft_tokens[:, :min_n_accepted]], dim=-1)

        if idx.shape[1] >= T + max_new_tokens:
            break

        if min_n_accepted < gamma:
            p_target_last = target_probs[:, min_n_accepted, :]
            corrected_token = torch.multinomial(p_target_last, num_samples=1)
            idx = torch.cat([idx, corrected_token], dim=-1)
        
        else:
            bonus_probs = F.softmax(full_target_logits[:, -1, :], dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            idx = torch.cat((idx, bonus_token), dim=1)

        return idx[:, :T + max_new_tokens]            


def universal_speculative_decode(
    target_model, target_decode, target_encode,
    draft_model, draft_decode, draft_encode,
    prompt_strings, max_new_tokens, gamma, eps=1e-12
):
    """
    Performs Universal Assisted Generation (UAG) for models with different tokenizers
    on a batch of prompts, using probabilistic acceptance.
    """
    target_model.eval()
    draft_model.eval()
    device = target_model.device

    # Initial encoding and padding for the target model
    encoded_prompts = [torch.tensor(target_encode(p), dtype=torch.long, device=device) for p in prompt_strings]
    max_len = max(p.shape[0] for p in encoded_prompts)
    idx = torch.stack([F.pad(p, (0, max_len - p.shape[0])) for p in encoded_prompts])
    
    B, T = idx.shape

    while idx.shape[1] < T + max_new_tokens:
        # --- 1. DRAFTING ---
        current_texts = [target_decode(seq.tolist()) for seq in idx]
        draft_encoded = [torch.tensor(draft_encode(t), dtype=torch.long, device=device) for t in current_texts]
        max_draft_len = max(p.shape[0] for p in draft_encoded)
        draft_idx_batch = torch.stack([F.pad(p, (0, max_draft_len - p.shape[0])) for p in draft_encoded])

        # Manual drafting to get draft tokens in their own vocab space
        draft_tokens_list = []
        draft_context = draft_idx_batch
        for _ in range(gamma):
            draft_logits, _ = draft_model(draft_context)
            next_draft_logits = draft_logits[:, -1, :]
            next_draft_probs = F.softmax(next_draft_logits, dim=-1)
            next_draft_token = torch.multinomial(next_draft_probs, num_samples=1)
            draft_tokens_list.append(next_draft_token)
            draft_context = torch.cat([draft_context, next_draft_token], dim=-1)
        draft_tokens_draft_space = torch.cat(draft_tokens_list, dim=-1)

        # --- 2. VERIFICATION PREP ---
        # Decode/re-encode each draft snippet to get them into the target's vocab space
        target_space_drafts = []
        for i in range(B):
            draft_snippet_text = draft_decode(draft_tokens_draft_space[i].tolist())
            target_space_drafts.append(torch.tensor(target_encode(draft_snippet_text), dtype=torch.long, device=device))
        
        min_gamma_target_space = min(len(t) for t in target_space_drafts)
        if min_gamma_target_space == 0: break 

        draft_tokens_in_target_space = torch.stack([t[:min_gamma_target_space] for t in target_space_drafts])

        # --- 3. VERIFICATION (Get p_target and p_draft) ---
        verify_input = torch.cat([idx, draft_tokens_in_target_space], dim=-1)
        full_target_logits, _ = target_model(verify_input)
        
        start_pos = idx.shape[1]
        target_logits = full_target_logits[:, start_pos-1:-1, :]
        target_probs = F.softmax(target_logits, dim=-1)
        p_target = torch.gather(target_probs, -1, draft_tokens_in_target_space.unsqueeze(-1)).squeeze(-1)

        # To get p_draft for the target-space tokens, we must re-evaluate the draft model
        p_draft_target_space_list = []
        for i in range(B):
            p_draft_sequence = []
            text_context_for_draft = current_texts[i]
            for j in range(min_gamma_target_space):
                draft_context_for_prob = torch.tensor([draft_encode(text_context_for_draft)], device=device)
                draft_logits, _ = draft_model(draft_context_for_prob)
                draft_probs_dist = F.softmax(draft_logits[:, -1, :], dim=-1)
                
                next_target_token_text = target_decode([draft_tokens_in_target_space[i, j].item()])
                # This can be tricky if tokenization doesn't align. We take the first token.
                next_draft_token_id = draft_encode(next_target_token_text)[0]
                
                prob = draft_probs_dist[0, next_draft_token_id]
                p_draft_sequence.append(prob)
                text_context_for_draft += next_target_token_text
            p_draft_target_space_list.append(torch.stack(p_draft_sequence))
        
        p_draft = torch.stack(p_draft_target_space_list)

        # --- 4. ACCEPTANCE (Probabilistic) ---
        alpha = (p_target / (p_draft + eps)).clamp(max=1.0)
        u = torch.rand_like(alpha)
        accepted_mask = (u <= alpha)
        prefix_mask = accepted_mask.int().cumprod(dim=-1).bool()
        min_n_accepted = prefix_mask.sum(dim=-1).min().item()

        # --- 5. UPDATE ---
        if min_n_accepted > 0:
            idx = torch.cat([idx, draft_tokens_in_target_space[:, :min_n_accepted]], dim=-1)
        
        if idx.shape[1] >= T + max_new_tokens: break

        if min_n_accepted < min_gamma_target_space:
            corrected_token_probs = target_probs[:, min_n_accepted, :]
            corrected_token = torch.multinomial(corrected_token_probs, num_samples=1)
            idx = torch.cat([idx, corrected_token], dim=-1)
        else:
            bonus_token_logits = full_target_logits[:, -1, :]
            bonus_token_probs = F.softmax(bonus_token_logits, dim=-1)
            bonus_token = torch.multinomial(bonus_token_probs, num_samples=1)
            idx = torch.cat([idx, bonus_token], dim=-1)
            
        print(f"UAG: Accepted {min_n_accepted}/{min_gamma_target_space} tokens for the batch.")

    final_tokens = idx[:, :T + max_new_tokens]
    return [target_decode(seq.tolist()) for seq in final_tokens]

# -----------------------------------------------------------------------------
# KL Divergence Loss
# -----------------------------------------------------------------------------
class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction="batchmean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, teacher_logits, student_logits):
        teacher_logits = teacher_logits / self.temperature
        student_logits = student_logits / self.temperature
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        loss = F.kl_div(student_log_probs, teacher_log_probs, reduction='none', log_target=True)
        loss = loss.sum(dim=-1)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "batchmean":
            return loss.sum() / teacher_logits.shape[0]
        else: # 'none'
            return loss






        