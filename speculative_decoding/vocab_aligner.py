import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class DraftModelVocabAligned(nn.Module):
    def __init__(self, TargetModel, DraftModel, freeze_draft: bool=False):
        super().__init__()
        self.embedder = DraftModel.embedder
        self.blocks = DraftModel.blocks

        emb_dim_draft = DraftModel.embedder.tok_wrd_embed.embedding_dim
        vocab_size_target = TargetModel.lm_head.out_features

        self.head = nn.Linear(emb_dim_draft, vocab_size_target)

        if freeze_draft:
            self.embedder.eval()
            self.blocks.eval()
            for p in self.embedder.parameters():
                p.requires_grad=False
            for p in self.blocks.parameters():
                p.requires_grad=False
    def forward(self, idx):
        x = self.embedder(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits, None


