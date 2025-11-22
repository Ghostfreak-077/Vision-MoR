import torch
import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTPreTrainedModel, ViTConfig, ViTEmbeddings, ViTPooler, ViTPatchEmbeddings, ViTLayer
from transformers.utils import auto_docstring, TransformersKwargs
from transformers.utils.generic import check_model_inputs
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from typing import Optional, Unpack


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

class ViTEncoder(nn.Module):
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

class MoRViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        """
        super().__init__(config)
        self.config = config

        self.num_recursions = 6

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, head_mask=head_mask)

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)

class MoRVisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
                 num_recursions=3, use_kv_sharing=False):
        super().__init__()
        self.num_recursions = num_recursions
        self.use_kv_sharing = use_kv_sharing

        # Initial embedding
        # self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

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

