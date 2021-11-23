from collections import OrderedDict

from catalyst.utils import load_checkpoint, unpack_checkpoint

from .base_config_initializer import BaseConfigInitializer
from .gpt2 import GPT2Model
from .bi_gpt2 import BiGPTModel
# from .wandb_logger import WandbLogger


class GPT2ConfigInitializer(BaseConfigInitializer):
    def init_model(self):
        if self.config.TYPE_MODEL == 'GPT2':
            model = GPT2Model(
                vocab_size=self.config.VOCAB_SIZE * 1000,
                sequence_length=self.config.SEQUENCE_LENGTH,
                head_size=self.config.HEAD_SIZE,
                n_layers=self.config.N_LAYERS,
                n_heads=self.config.N_HEADS,
            )
        elif self.config.TYPE_MODEL == 'BiGPT2':
            model = BiGPTModel(
                vocab_size=self.config.VOCAB_SIZE * 1000,
                sequence_length=self.config.SEQUENCE_LENGTH,
                hidden_size=self.config.HEAD_SIZE,
                n_layers=self.config.N_LAYERS,
                n_heads=self.config.N_HEADS,
            )
        else:
            raise AttributeError('Please set correct attribute "TYPE_MODEL" in config')
        return model

    @staticmethod
    def _create_checkpoint_from_adaptive(checkpoint):
        crop_state_dict = OrderedDict()
        for key, weight in checkpoint['model_state_dict'].items():
            if key.startswith('gpt'):
                crop_state_dict[f'{key[4:]}'] = weight
        return crop_state_dict

    def _unpack_checkpoint(self, model, criterion, optimizer):
        if self.config.CHECKPOINT_PATH is not None:
            checkpoint = load_checkpoint(self.config.CHECKPOINT_PATH)
            model = model.cuda()

            checkpoint_model_type = (
                self.config.MODEL_ADDITIONAL_ARGUMENTS.get('checkpoint_model_type', None)
            )
            if checkpoint_model_type == 'adaptive':
                crop_state_dict = self._create_checkpoint_from_adaptive(checkpoint)
                model.gpt.transformer.load_state_dict(crop_state_dict)
            elif checkpoint_model_type is None:
                model = model.cuda()
                unpack_checkpoint(
                    checkpoint,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                )
            else:
                raise TypeError('unknown checkpoint_model_type')
        return model
