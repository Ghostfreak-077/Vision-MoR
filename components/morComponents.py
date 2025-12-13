import torch
import torch.nn as nn

class MoRRouter(nn.Module):
    """Expert-choice router with auxiliary loss"""

    def __init__(self, embed_dim, num_recursions):
        super().__init__()
        self.num_recursions = num_recursions
        self.router = nn.Linear(embed_dim, 1)

    def forward(self, x, recursion_step, capacity_factor):
        """
        x: (B, N, C) input tokens
        recursion_step: current recursion depth (0-indexed)
        capacity_factor: fraction of tokens to select
        """
        B, N, C = x.shape

        # Compute routing scores
        scores = torch.sigmoid(self.router(x)).squeeze(-1)  # (B, N)

        # Select top-k tokens based on capacity
        k = max(1, int(N * capacity_factor))
        top_scores, top_indices = torch.topk(scores, k, dim=1)

        # Create selection mask
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)

        return mask, scores

