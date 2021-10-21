from collections import OrderedDict

from catalyst.utils import load_checkpoint, unpack_checkpoint

from .base_config_initializer import BaseConfigInitializer
from .gpt2 import GPT2Model
from .wandb_logger import WandbLogger


class GPT2ConfigInitializer(BaseConfigInitializer):
    def init_model(self):
        model = GPT2Model(
            vocab_size=self.config.VOCAB_SIZE * 1000,
            sequence_length=self.config.SEQUENCE_LENGTH,
            head_size=self.config.HEAD_SIZE,
            n_layers=self.config.N_LAYERS,
            n_heads=self.config.N_HEADS,
        )

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

    def init_logging_callback(self, logdir):
        config = self.config

        wandb_logger = WandbLogger(
            log_on_batch_end=True,
            log_on_epoch_end=True,
            log_each_n_batch=100,
            project='rr',
            group=config.WANDB_GROUP if hasattr(config, 'WANDB_GROUP') else 'test',
            name=config.model_name,
            config={
                'model_name': config.MODEL_NAME,
                'sequence_length': config.SEQUENCE_LENGTH,
                'tokenizer': config.TOKENIZER,
                'num_epochs': config.N_EPOCH,
                'batch_size': config.BATCH_SIZE,
                'h_size': config.HEAD_SIZE,
                'n_heads': config.N_HEADS,
                'vocab_size': config.VOCAB_SIZE,
                'n_layers': config.N_LAYERS,
                'dropout': config.DROPOUT,
                'optim': (
                    config.OPTIMIZER_NAME if hasattr(config, 'OPTIMIZER_NAME')
                    else str(config.OPTIMIZER_CLASS)
                ),
                **{
                    f'optim_{key}': f'{value}'
                    for key, value in config.OPTIMIZER_ADDITIONAL_ARGUMENTS.items()
                },
                'max_norm': config.MAX_NORM,
                'accumulation_steps': config.ACCUMULATION_STEPS,
                'scheduler': (
                    config.SCHEDULER_NAME if hasattr(config, 'SCHEDULER_NAME')
                    else str(config.SCHEDULER_CLASS)
                ),
                **{
                    f'scheduler_{key}': f'{value}'
                    for key, value in config.SCHEDULER_ADDITIONAL_ARGUMENTS.items()
                },
                'distributed_mode': config.use_distributed_mode,
                'config_name': config.config_name,
                'logdir': logdir,
                'additional': '',
                'num_workers': config.NUM_WORKERS,
            }
        )

        return wandb_logger