import transformers
from torch.nn.functional import log_softmax

from ..utils.technical import annotations_from_parent
from .base_model import BaseModel


@annotations_from_parent
class GPT2Model(BaseModel):
    def __init__(
            self,
            vocab_size: int,
            sequence_length: int,
            head_size: int,
            n_layers: int,
            n_heads: int,
            is_raw_output: bool = False,
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
        is_raw_output : bool
            If True then use as output raw output of transformer gpt.
        """
        super(GPT2Model, self).__init__()
        gpt2_config = transformers.GPT2Config(
            vocab_size=vocab_size,
            n_positions=sequence_length,
            n_ctx=sequence_length,
            n_embd=head_size,
            n_layer=n_layers,
            n_head=n_heads,
        )
        self.gpt = transformers.GPT2LMHeadModel(gpt2_config)
        self.gpt.lm_head.weight = self.gpt.transformer.wte.weight

        self._sequence_length = sequence_length
        self.is_raw_output = is_raw_output

    @property
    def max_context_length(self):
        return self.gpt.config.n_ctx

    def forward(self, x):
        x = self.gpt(x)
        x[0].transpose_(1, 2)
        if self.is_raw_output:
            return x

        return x[0]

    def get_next_token_scores(self, input_ids, attention_mask=None, past=None, use_cache=None):
        # only last token is needed
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)         

        transformer_outputs = self.gpt.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past=past,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]

        # only last token output goes to the linear layer
        logits = self.gpt.lm_head(hidden_states[:, -1, :])

        if use_cache and len(transformer_outputs) > 1:
            new_past = transformer_outputs[1]
        else:
            new_past = None

        return logits, new_past


def load_weights_from_adaptive_to_base(destination_model, adaptive_model):
    apadtive_model_gpt_state_dict = adaptive_model.gpt.state_dict()
    destination_model.gpt.transformer.load_state_dict(apadtive_model_gpt_state_dict)

