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









        