from copy import deepcopy
from functools import partial
from typing import Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from ..utils.technical import annotations_from_parent
from .base_model import BaseModel


def run_right_gpt(reverted_input_tensor, model):
    right_to_left_output = model.forward(reverted_input_tensor)
    right_to_left_reverted_back_output = torch.flip(
        right_to_left_output[0],
        dims=(1,)
    )
    return right_to_left_reverted_back_output


def run_right_cnn(reverted_input_tensor, model):
    features = model(reverted_input_tensor)
    return features


def wrapper_run_right_embedding(wte_model, position_emb, seq_len=512):
    indexes = list(range(-seq_len + 1, seq_len))

    pos_ind = [
        [indexes[1022 - i - j] for i in range(seq_len)]
        for j in range(seq_len)
    ]
    pos_indexes = torch.tensor(pos_ind, dtype=torch.long)
    pos_indexes[pos_indexes < 0] = 0

    mask_ = [
        [i <= j for i in range(seq_len)]
        for j in range(seq_len - 1, -1, -1)
    ]
    mask = torch.tensor(mask_, dtype=torch.long)

    n_count = torch.sum(mask, dim=0).unsqueeze(0).unsqueeze(0)

    def run_right_embedding(reverted_input_tensor):
        device = reverted_input_tensor.device

        tokens = wte_model(reverted_input_tensor).permute(0, 2, 1)  # [batch, hid_dim, seq_len]

        pos_index = pos_indexes.to(device)  # [seq_len]
        pos_emb = position_emb(pos_index).squeeze(2) * mask.to(device)  # [seq_len, seq_len]

        features = (tokens @ pos_emb) / n_count.to(device)

        return torch.flip(features, dims=(1,))

    return run_right_embedding


class CNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            n_filters,
            filter_sizes,
            padding=0,
            dropout=0.3,
            depthwise=False
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList()
        for filter_size in filter_sizes:
            if depthwise:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=1),
                    nn.Conv2d(
                        in_channels=n_filters,
                        out_channels=n_filters,
                        kernel_size=(filter_size, embedding_dim),
                        padding=(padding, 0),
                        groups=n_filters,
                    )
                )
            else:
                conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, embedding_dim))
            self.convs.append(conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        many_conved = []
        for conv in self.convs:
            many_conved.append(
                F.relu(self.dropout(conv(embedded)).squeeze(3))
            )
        pooled = []
        for conved in many_conved:
            pooled.append(
                F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            )
        cat = self.dropout(torch.cat(pooled, dim=1))
        return cat


# TODO: return back annotations
# delete annotations because it raise exception
# TypeError: different varnames forward: 
# parent varnames: ['self', 'x'], 
# child varnames: ['self', 'input_tensor', 'reverted_input_tensor']
class BiGPTModel(BaseModel):
    def __init__(
            self,
            vocab_size: int,
            sequence_length: int,
            head_size: int,
            n_layers: int,
            n_heads: int,
            dropout: float,
            right_model_type: str,
            right_model_config: Any,
            stack_right_left: bool = None,
            one_wpe: bool = None,
            one_wte: bool = None,
            lm_equal_emb: bool = None,
    ):
        """

        Parameters
        ----------
        vocab_size : int
            Vocabulary size.
        sequence_length : int
            Sequence max length.
        head_size : int
        n_layers : int
        n_heads : int
        dropout: float
        right_model_params: dataclass with parameters
        """
        super(BiGPTModel, self).__init__()

        left_params = {
            'vocab_size': vocab_size,
            'n_positions': sequence_length,
            'n_ctx': sequence_length,
            'n_embd': head_size,
            'n_layer': n_layers,
            'n_head': n_heads,
            'resid_pdrop': dropout,
            'embd_pdrop': dropout,
            'attn_pdrop': dropout,
        }
        left_gpt2_config = transformers.GPT2Config(**left_params)
        self.gpt_left_to_right = transformers.GPT2Model(left_gpt2_config)
        right_params = deepcopy(left_params)
        if right_model_type.value == 'GPT2':
            self.gpt_right_to_left = self.create_right_gpt(right_params, right_model_config)
            self.forward_right_context = partial(run_right_gpt, model=self.gpt_right_to_left)
        elif right_model_type.value == 'EMB':
            self.gpt_right_to_left = nn.Embedding(
                right_model_config.NUM_EMBEDDINGS, right_model_config.EMBEDDING_DIM
            )
            self.forward_right_context = wrapper_run_right_embedding(
                wte_model=self.gpt_left_to_right.wte, position_emb=self.gpt_right_to_left
            )
        elif right_model_type.value == 'CNN':
            self.gpt_right_to_left = self.create_right_cnn(right_params, right_model_config)
            self.forward_right_context = partial(run_right_cnn, model=self.gpt_right_to_left)
        else:
            raise ValueError('Strange right model type')

        if stack_right_left:
            self.lm_head = nn.Sequential(
                nn.Linear(right_params['n_embd'] + left_params['n_embd'], head_size),
                nn.Linear(head_size, vocab_size),
            )
        else:
            self.lm_head = nn.Linear(right_params['n_embd'] + left_params['n_embd'], vocab_size)

        if one_wpe:
            self.gpt_right_to_left.wpe = self.gpt_left_to_right.wpe
        if one_wte and right_params['n_embd'] == left_params['n_embd']:
            self.gpt_right_to_left.wte = self.gpt_left_to_right.wte
        if stack_right_left and lm_equal_emb:
            self.lm_head.weight = self.gpt_left_to_right.wte.weight

        self._sequence_length = sequence_length

    @property
    def max_context_length(self):
        return self.gpt_left_to_right.config.n_ctx

    @staticmethod
    def create_right_gpt(gpt_config, right_model_config):
        if right_model_config.DROPOUT is not None:
            gpt_config['resid_pdrop'] = right_model_config.DROPOUT
            gpt_config['embd_pdrop'] = right_model_config.DROPOUT
            gpt_config['attn_pdrop'] = right_model_config.DROPOUT
        if right_model_config.HEAD_SIZE is not None:
            gpt_config['n_embd'] = right_model_config.HEAD_SIZE
        return transformers.GPT2Model(transformers.GPT2Config(**gpt_config))

    @staticmethod
    def create_right_cnn(gpt_config, right_model_config):
        model = CNN(gpt_config['vocab_size'], gpt_config['n_embd'], 16, [1, 2, 3, 3, 4, 4, 5, 6])
        return model

    def forward(
            self,
            input_tensor: torch.Tensor,
            reverted_input_tensor: torch.Tensor,
    ):
        """
        Input_tensor and reverted_tensor should be shifted before the forward!

        Parameters
        ----------
        input_tensor : left context of model
        reverted_input_tensor : right context of model
        """
        # apply each of the network separately
        left_to_right_output = self.gpt_left_to_right.forward(input_tensor)
        right_to_left_reverted_back_output = self.forward_right_context(reverted_input_tensor)

        # we need to revert right-to-left network before the concat

        # concat both network outputs and apply lm head,
        concatenate_outputs = torch.cat(
            (left_to_right_output[0], right_to_left_reverted_back_output),
            dim=2,
        )
        logits = self.lm_head(concatenate_outputs)

        # permute output from [batch, seq_len, vocab] to [batch, vocab, seq_len]
        # it needed for nn.CrossEntropyLoss
        logits = logits.permute(0, 2, 1)

        return logits

    def forward_without_reverted_tensor(
            self, input_tensor: torch.Tensor, right_to_left_shift: int = 2
    ):
        assert input_tensor.size(1) <= right_to_left_shift
        reverted_input_tensor = torch.flip(input_tensor, dims=(1,))

        # apply each of the network separately
        left_to_right_output = self.gpt_left_to_right.forward(input_tensor)
        right_to_left_output = self.gpt_right_to_left.forward(
            reverted_input_tensor
        )

        # we need to revert right-to-left network before the concat
        right_to_left_reverted_back_output = torch.flip(
            right_to_left_output[0],
            dims=(1,)
        )

        # shift each of the network
        shifted_left_to_right_output = (
            left_to_right_output[0][:, :-right_to_left_shift, :]
        )
        shifted_right_to_left_output = (
            right_to_left_reverted_back_output[:, right_to_left_shift:, :]
        )

        # concat both network outputs and apply lm head
        concatenate_outputs = torch.cat(
            (shifted_left_to_right_output, shifted_right_to_left_output),
            dim=2,
        )
        logits = self.lm_head(concatenate_outputs)
        # permute output from [batch,seq_len,vocab] to [batch,vocab,seq_len]
        # it needed for nn.CrossEntropyLoss
        logits = logits.permute(0, 2, 1)
        return logits

    @torch.no_grad()
    def get_next_token_scores(
            self,
            input_ids: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Tuple[
                Union[torch.Tensor, None], Union[torch.Tensor, None]
            ] = (None, None),
            past: Optional[Tuple[Union[torch.Tensor, None]]] = (None, None),
            use_cache: Optional[bool] = None,
    ):
        # all parameters be a tuple for two net
        input_left_to_right, input_right_to_left = input_ids

        attention_left_to_right, attention_right_to_left = attention_mask
        past_left_to_right, past_right_to_left = past

        if past_left_to_right is not None:
            input_left_to_right = input_left_to_right[:, -1].unsqueeze(-1)

        if past_right_to_left is not None:
            input_right_to_left = input_right_to_left[:, -1].unsqueeze(-1)

        # forward left to right model
        output_left_to_right = self.gpt_left_to_right(
            input_ids=input_left_to_right,
            attention_mask=attention_left_to_right,
            past=past_left_to_right,
            use_cache=use_cache
        )

        hidden_left_to_right = output_left_to_right[0][:, -1, :]

        # forward right to left model        
        output_right_to_left = self.forward_right_context(input_right_to_left)
        hidden_right_to_left = output_right_to_left[0][:, -1, :]

        hidden_states = torch.cat(
            (
                hidden_left_to_right,
                hidden_right_to_left.repeat(hidden_left_to_right.size(0), 1)
            ),
            dim=1,
        )

        logits = self.lm_head(hidden_states)

        if use_cache:
            # return caching right context
            new_past = (output_left_to_right[1], None)
        else:
            new_past = (None, None)

        return logits, new_past
