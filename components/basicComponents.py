import torch
import torch.nn as nn
from components.morComponents import MoRRouter
from transformers.models.vit.modeling_vit import ViTConfig, ViTLayer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional

class MoRViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_recursions = config.num_recursions
        self.routers = nn.ModuleList([
            MoRRouter(config.hidden_size, config.num_recursions)
            for _ in range(config.num_recursions)
        ])
        self.first_block = ViTLayer(config)
        self.last_block = ViTLayer(config)
        blocks_per_recursion = (config.num_hidden_layers - 2) // config.num_recursions

        if blocks_per_recursion <= 0:
            raise ValueError("num_hidden_layers is too small for the given num_recursion.")

        print(f"Blocks per recursion: {blocks_per_recursion}")
        print(f"num_hidden_layers: {config.num_hidden_layers}")
        print(f"num_recursion: {config.num_recursions}")
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(blocks_per_recursion)])
        self.gradient_checkpointing = False
        self.register_buffer('capacity_factors',
                             torch.tensor([1.0 - i / config.num_recursions for i in range(config.num_recursions)]))

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:

        x = self.first_block(hidden_states)

        routing_decisions = []
        for idx in range(self.num_recursions):

            mask, scores = self.routers[idx](x, idx, self.capacity_factors[idx])
            routing_decisions.append(mask)

            for i, layer_module in enumerate(self.layer):
                # layer_head_mask = head_mask[i] if head_mask is not None else None
                if len(mask.shape) < 4:
                    mask = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N)
                x = layer_module(x, mask)

        last_hidden_state = self.last_block(x)

        return BaseModelOutput(last_hidden_state=last_hidden_state)
