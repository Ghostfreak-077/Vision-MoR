import torch
from torch import nn
from components.basicComponents import PatchEmbedding, MultiHeadAttention, MLP, TransformerBlock

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


class MoRRecursiveBlock(nn.Module):
    """Shared transformer block for recursion"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.block = TransformerBlock(embed_dim, num_heads, mlp_ratio)
        
    def forward(self, x, mask=None, kv_cache=None):
        """
        x: (B, N, C)
        mask: (B, N) boolean mask for active tokens
        kv_cache: optional KV cache
        """
        if mask is not None:
            # Only process selected tokens
            B, N, C = x.shape
            active_tokens = x[mask].view(B, -1, C)
            active_out, kv = self.block(active_tokens, kv_cache)
            
            # Scatter back to original positions
            out = x.clone()
            out[mask] = active_out.view(-1, C)
            return out, kv
        else:
            return self.block(x, kv_cache)