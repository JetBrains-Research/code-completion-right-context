from collections import OrderedDict

from catalyst.utils import load_checkpoint, unpack_checkpoint

from .base_config_initializer import BaseConfigInitializer
from .gpt2 import GPT2Model


class GPT2ConfigInitializer(BaseConfigInitializer):

    def init_model(self):
        model = GPT2Model(
            vocab_size=self.config.VOCAB_SIZE * 1000,
            sequence_length=self.config.SEQUENCE_LENGTH,
            head_size=self.config.HEAD_SIZE,
            n_layers=self.config.N_LAYERS,
            n_heads=self.config.N_HEADS,
            dropout=self.config.DROPOUT,
        )
        return model
