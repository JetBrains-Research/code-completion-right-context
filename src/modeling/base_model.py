from torch import nn as nn

import torch
from typing import Iterable, Tuple


class BaseModel(nn.Module):
    """
    Abstract class for all dl models.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_next_token_scores(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            past: Iterable[torch.Tensor] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Get log probabilities for next token.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token indexes of input sequence.
            Tensor of shape (1, sequence_length).
        attention_mask : torch.Tensor
            Attention mask for input sequence.
            By default no mask is used.
        past : list of torch.Tensor
            Previous outputs for all layers.
            It is used only if use_cache is True.
            Each list element is one layer outputs.
            See transformers.GPT2LMHeadModel.__call__ past argument for details.
        use_cache : bool
            If True then cache from `past` is used.
            Cache using increases the generation speed.

        Returns
        -------
        next_token_scores : torch.Tensor
            Next token log probabilities.
        new_past : list of torch.Tensor
            New value for the `past` variable (all outputs for all layers).
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """
        Get device for model.
        Attention!
        If model is stored on several devices
        this method works wrong.
        """
        some_weights = next(iter(self.parameters()))
        return some_weights.device
