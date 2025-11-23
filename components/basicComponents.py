import torch
import torch.nn as nn
from components.morComponents import MoRRouter
from transformers.models.vit.modeling_vit import ViTConfig, ViTLayer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional

class MoRViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig, num_recursion=6):
        super().__init__()
        self.config = config
        self.num_recursion = num_recursion
        self.routers = nn.ModuleList([
            MoRRouter(config.hidden_size, num_recursion)
            for _ in range(num_recursion)
        ])
        self.first_block = ViTLayer(config)
        self.last_block = ViTLayer(config)
        blocks_per_recursion = (config.num_hidden_layers - 2) // self.num_recursion
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(blocks_per_recursion)])
        self.gradient_checkpointing = False
        self.register_buffer('capacity_factors',
                             torch.tensor([1.0 - i / num_recursion for i in range(num_recursion)]))

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:

        x = self.first_block(hidden_states)

        routing_decisions = []
        for idx in range(self.num_recursion):

            mask, scores = self.routers[idx](x, idx, self.capacity_factors[idx])
            routing_decisions.append(mask)

            for i, layer_module in enumerate(self.layer):
                # layer_head_mask = head_mask[i] if head_mask is not None else None
                x = layer_module(x, mask)

        last_hidden_state = self.last_block(x)

        return BaseModelOutput(last_hidden_state=last_hidden_state)
