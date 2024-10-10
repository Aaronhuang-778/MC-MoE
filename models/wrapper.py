import itertools as I
import logging
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import pickle
import numpy as np

from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MixtralBLockSparseTop2MLP,
    MixtralRMSNorm,
    MISTRAL_ATTENTION_CLASSES)

from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from data.cachedata import CacheDataset
from utils.normal_quantizer import normal_quantize

logger = logging.getLogger(__name__)


class PrunableMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 r: Optional[int] = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r
        self.num_experts = model.num_experts
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.weights_tensor = torch.zeros_like(torch.zeros(8, device='cuda:0'))
        self.number_tensor = torch.zeros_like(torch.zeros(8, device='cuda:0'))

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.to(
            device=self.model.gate.weight.data.device, non_blocking=True)
        router_logits = self.model.gate(hidden_states)
        
        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)
        
        expert_weights_sum = torch.zeros(self.num_experts, device=hidden_states.device)
        expert_weights_sum = expert_weights_sum.scatter_add(0, selected_experts.view(-1), routing_weights.view(-1))

        self.weights_tensor = self.weights_tensor.to(device=hidden_states.device)
        self.weights_tensor += expert_weights_sum
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=hidden_states.device)
        ones = torch.ones_like(selected_experts.view(-1), device=hidden_states.device)
        expert_counts = expert_counts.scatter_add(0, selected_experts.view(-1), ones)
        self.number_tensor = self.number_tensor.to(device=hidden_states.device)
        self.number_tensor += expert_counts

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warn(
                f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    @torch.no_grad()
    def enumerate(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        loss_history = dict()

        with torch.inference_mode():
            for dropped in I.combinations(range(self.model.num_experts), self.model.num_experts - self.r):
                self.experts_to_drop = dropped
                loss = 0
                for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):
                    hidden_states = hidden_states.to(
                        device=self.model.gate.weight.data.device, non_blocking=True)
                    final_hidden_states = final_hidden_states.to(
                        dtype=torch.float64, device=self.model.gate.weight.data.device, non_blocking=True)

                    final_hidden_states_e, _ = self.forward(
                        hidden_states.unsqueeze(0))
                    loss += torch.norm(final_hidden_states -
                                       final_hidden_states_e.squeeze(0).to(torch.float64)).item()
                loss_history[dropped] = loss

        self.experts_to_drop = min(loss_history, key=loss_history.get)
        return loss_history
    

    @torch.no_grad()
    def prune(self):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop))

        gate_new = torch.nn.Linear(in_features=self.model.gate.in_features,
                                   out_features=self.r, bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(
            experts_to_reserve)]
        self.model.gate = gate_new

        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in experts_to_reserve])
        self.model.num_experts = self.r

class QuantbleMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 r: Optional[int] = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r
        self.num_experts = model.num_experts
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.wbits = 0

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.to(
            device=self.model.gate.weight.data.device, non_blocking=True)
        router_logits = self.model.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, routing_weights[top_x_list, idx_list, None])
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    @torch.no_grad()
    def order_quant(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        loss_history = dict()

        with torch.inference_mode():
            for expert_id in range(self.model.num_experts):
                w1 = self.model.experts[expert_id].w1.weight.data.clone()
                w2 = self.model.experts[expert_id].w2.weight.data.clone()
                w3 = self.model.experts[expert_id].w3.weight.data.clone()
                quant_loss = dict()
                for wbit in [1, 2, 3]:
                    self.wbits = wbit 
                    loss = 0
                    # quant with wbit
                    self.model.experts[expert_id].w1.weight.data = normal_quantize(w=self.model.experts[expert_id].w1.weight.data, blocksize=128, wbit=wbit)
                    self.model.experts[expert_id].w2.weight.data = normal_quantize(w=self.model.experts[expert_id].w2.weight.data, blocksize=128, wbit=wbit)
                    self.model.experts[expert_id].w3.weight.data = normal_quantize(w=self.model.experts[expert_id].w3.weight.data, blocksize=128, wbit=wbit)
                    for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):
                        hidden_states = hidden_states.to(
                            device=self.model.gate.weight.data.device, non_blocking=True)
                        final_hidden_states = final_hidden_states.to(
                            dtype=torch.float64, device=self.model.gate.weight.data.device, non_blocking=True)
                        final_hidden_states_e, _ = self.forward(
                            hidden_states.unsqueeze(0))
                        # back to fp16
                        loss += torch.norm(final_hidden_states -
                                final_hidden_states_e.squeeze(0).to(torch.float64)).item()
                    self.model.experts[expert_id].w1.weight.data = w1.clone()
                    self.model.experts[expert_id].w2.weight.data = w2.clone()
                    self.model.experts[expert_id].w3.weight.data = w3.clone()
                    quant_loss[wbit] = loss

                loss_history[expert_id] = quant_loss
        return loss_history
    

    @torch.no_grad()
    def prune(self):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop))

        gate_new = torch.nn.Linear(in_features=self.model.gate.in_features,
                                   out_features=self.r, bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(
            experts_to_reserve)]
        self.model.gate = gate_new

        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in experts_to_reserve])
        self.model.num_experts = self.r



class DynamicRankMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 r: Optional[int] = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r
        self.num_experts = self.model.num_experts
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.block_index = 0
        self.loss_index = 0
        self.beta = 0.5
        self.tau_l = 0.1
        self.tau_h = 0.1
        self.block_range = 20

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor, self_attn_weights: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        #router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.to(
            device=self.model.gate.device, non_blocking=True)
        router_logits = self.model.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        ###################selected metric###################

        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)
        
        if self.loss_index < self.block_range:
            # 2. use the input attention weights to lower the rank
            new_routing = torch.ones_like(routing_weights)

            mask_top1 = (routing_weights[:, 1] <  self.beta * routing_weights[:, 0])
            new_routing[mask_top1, 1] = 0
            
            if sequence_length > 1:
                mean_tensor = torch.mean(self_attn_weights.squeeze(), dim=0)
                if len(mean_tensor.shape) == 2:
                    row, column = mean_tensor.shape
                    real_number = torch.arange(column, 0, -1).to(device=mean_tensor.device)
                    attention_scores = torch.sum(mean_tensor, dim=1) / real_number
                else:
                    attention_scores = mean_tensor
                l1_norm = torch.norm(hidden_states, p=1, dim=1).to(device=attention_scores.device)
                attention_scores = attention_scores * l1_norm
                # diagonal_elements = torch.diag(mean_tensor)
                _, mask_all = torch.topk(attention_scores, int(self.tau_l * len(attention_scores)), largest=False)
                new_routing[mask_all, :] = 0

                _, keep_all = torch.topk(attention_scores, int(self.tau_h * len(attention_scores)), largest=True)
                new_routing[keep_all, :] = 1

            routing_weights = routing_weights * new_routing

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts)
        
        if self.loss_index < self.block_range:
            experts_tmp = expert_mask.clone()
            expert_mask[mask_top1, 1, :] = 0
            if sequence_length > 1:
                expert_mask[mask_all, :, :] = 0
                expert_mask[keep_all, :, :] = experts_tmp[keep_all, :, :]
        expert_mask = expert_mask.permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, routing_weights[top_x_list, idx_list, None])
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits

class DynamicRankMixtralDecoderLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = True,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.model.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.model.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
        )
        residual = residual.to(
            device=hidden_states.device, non_blocking=True)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.model.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.model.block_sparse_moe(hidden_states, self_attn_weights)
        residual = residual.to(
            device=hidden_states.device, non_blocking=True)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs