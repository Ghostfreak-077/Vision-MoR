import torch
from torch import nn
from components.basicComponents import PatchEmbedding, MultiHeadAttention, MLP, TransformerBlock
from components.morComponents import MoRRouter, MoRRecursiveBlock

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x, _ = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        return self.head(cls_output)
    
class MoRVisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, 
                 num_recursions=3, use_kv_sharing=False):
        super().__init__()
        self.num_recursions = num_recursions
        self.use_kv_sharing = use_kv_sharing
        
        # Initial embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        # First layer (not shared)
        self.first_block = TransformerBlock(embed_dim, num_heads, mlp_ratio)
        
        # Shared recursive blocks (Middle-Cycle strategy)
        blocks_per_recursion = (depth - 2) // num_recursions
        self.shared_blocks = nn.ModuleList([
            MoRRecursiveBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(blocks_per_recursion)
        ])
        
        # Router for each recursion step
        self.routers = nn.ModuleList([
            MoRRouter(embed_dim, num_recursions)
            for _ in range(num_recursions)
        ])
        
        # Last layer (not shared)
        self.last_block = TransformerBlock(embed_dim, num_heads, mlp_ratio)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Capacity factors for each recursion (decreasing)
        self.register_buffer('capacity_factors', 
                           torch.tensor([1.0 - i/num_recursions for i in range(num_recursions)]))
        
    def forward(self, x):
        B = x.shape[0]
        N = self.patch_embed.num_patches + 1
        
        # Patch embedding
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # First non-shared block
        x, first_kv = self.first_block(x)
        
        # Store KV cache for sharing if enabled
        kv_cache = first_kv if self.use_kv_sharing else None
        
        # Recursive processing with routing
        routing_decisions = []
        for rec_idx in range(self.num_recursions):
            # Get routing decision
            mask, scores = self.routers[rec_idx](x, rec_idx, self.capacity_factors[rec_idx])
            routing_decisions.append(mask)
            
            # Apply shared blocks to selected tokens
            for block in self.shared_blocks:
                x, kv = block(x, mask, kv_cache if self.use_kv_sharing else None)
                if not self.use_kv_sharing:
                    kv_cache = kv
        
        # Last non-shared block
        x, _ = self.last_block(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        # Compute auxiliary loss for router training
        aux_loss = self.compute_auxiliary_loss(routing_decisions)
        
        return logits, aux_loss
    
    def compute_auxiliary_loss(self, routing_decisions):
        """Auxiliary loss to encourage binary routing decisions"""
        aux_loss = 0.0
        for mask in routing_decisions:
            # Convert boolean mask to float
            decisions = mask.float()
            # Encourage decisions towards 0 or 1
            aux_loss += torch.mean(decisions * (1 - decisions))
        return aux_loss * 0.01  # Small coefficient

