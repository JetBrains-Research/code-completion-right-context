from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import transformers

from ..utils.technical import annotations_from_parent
from .base_model import BaseModel


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
        """
        super(BiGPTModel, self).__init__()

        # both parts have the same amount of parameters
        gpt2_config = transformers.GPT2Config(
            vocab_size=vocab_size,
            n_positions=sequence_length,
            n_ctx=sequence_length,
            n_embd=head_size,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.gpt_left_to_right = transformers.GPT2Model(gpt2_config)
        self.gpt_right_to_left = transformers.GPT2Model(gpt2_config)
        self.lm_head = nn.Linear(head_size * 2, vocab_size)

        self._sequence_length = sequence_length

    @property
    def max_context_length(self):
        return self.gpt_left_to_right.config.n_ctx

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
        right_to_left_output = self.gpt_right_to_left.forward(
            reverted_input_tensor
        )

        # we need to revert right-to-left network before the concat
        right_to_left_reverted_back_output = torch.flip(
            right_to_left_output[0],
            dims=(1,)
        )

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
        output_right_to_left = self.gpt_right_to_left(
            input_ids=input_right_to_left,
            attention_mask=attention_right_to_left,
            past=past_right_to_left,
            use_cache=use_cache            
        )
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
