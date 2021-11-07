import os
import shutil

from catalyst import dl
from catalyst.utils import load_checkpoint, unpack_checkpoint
from catalyst import callbacks as catalyst_callbacks

from .dataset import DatasetLoaderInitializer


class BaseConfigInitializer:
    """
    Abstract class for model initialization.

    Use this class to initialize model from .py config.
    """
    def __init__(self, config):
        self.config = config

    def init_dataset_and_loaders(self):
        config = self.config
        is_bi_gpt = True if self.config.TYPE_MODEL == 'BiGPT2' else False
        initializer = DatasetLoaderInitializer(
            data_dir=config.DATA_DIR,
            tokenizer_name=config.TOKENIZER,
            vocab_size=config.VOCAB_SIZE,
            sequence_length=config.SEQUENCE_LENGTH,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            use_first_n_objects=config.use_first_n_objects,
            train_mode=config.DATASET_TRAIN_MODE,
            valid_mode=config.DATASET_VALID_MODE,
            is_bi_gpt=is_bi_gpt,
            **config.DATASET_ADDITIONAL_ARGUMENTS,
        )
        datasets, loaders = initializer.initialize_dataset_and_loaders()

        return datasets, loaders

    def reset_logdir(self):
        config = self.config
        logdir = (
            f'{config.HOME_DIR}/logs/'
            f'{config.WANDB_GROUP}_{config.model_name}'
        )
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            pass
        os.makedirs(logdir, exist_ok=True)

        return logdir

    def init_criterion(self):
        return self.config.CRITERION_CLASS()

    def init_optimizer_and_scheduler(self, model, loaders):
        """

        Parameters
        ----------
        model: BaseModel inherited
        loaders : list of DataLoader

        Returns
        -------
        optimizer : torch optimizer
        scheduler : torch scheduler
        """
        config = self.config
        optimizer = config.OPTIMIZER_CLASS(
            model.parameters(),
            **config.OPTIMIZER_ADDITIONAL_ARGUMENTS,
        )
        if config.SCHEDULER_CLASS is not None:
            scheduler = config.SCHEDULER_CLASS(
                optimizer=optimizer,
                epochs=config.N_EPOCH,
                steps_per_epoch=len(loaders['train']),
                **config.SCHEDULER_ADDITIONAL_ARGUMENTS,
            )
        else:
            scheduler = None

        return optimizer, scheduler

    def _init_base_callbacks(self):
        """
        Initialization of base callbacks, that are equal for all models.

        Returns
        -------
        callbacks : list of catalyst callbacks
        """
        config = self.config

        if hasattr(config, 'HOME_DIR'):
            saved_checkpoint_amount = config.HOME_DIR
        else:
            saved_checkpoint_amount = 3

        callbacks = [
            catalyst_callbacks.EarlyStoppingCallback(
                saved_checkpoint_amount,
                loader_key="valid",
                metric_key="loss",
                minimize="loss"
            ),
            catalyst_callbacks.CheckpointCallback(saved_checkpoint_amount),
            catalyst_callbacks.SchedulerCallback(
                loader_key="train", metric_key="loss"
            )
        ]

        if config.MAX_NORM is not None:
            optimizer_callback = catalyst_callbacks.OptimizerCallback(
                metric_key="loss",
                accumulation_steps=config.ACCUMULATION_STEPS,
#                 grad_clip_params={
#                     "func": "clip_grad_norm_",
#                     "max_norm": config.MAX_NORM,
#                     "norm_type": 2
#                 }
            )
        else:
            optimizer_callback = catalyst_callbacks.OptimizerCallback(
                metric_key="loss",
                accumulation_steps=config.ACCUMULATION_STEPS,
            )
        callbacks.append(optimizer_callback)

        if hasattr(config, 'SCHEDULER_MODE') and config.SCHEDULER_MODE is not None:
            for scheduler_key, scheduler_mode in config.SCHEDULER_MODE.items():
                callbacks.append(
                    catalyst_callbacks.SchedulerCallback(
                        loader_key="train", metric_key="loss"
                    )
                )

        return callbacks

    def init_callbacks(self, logdir, criterion=None):
        """

        Parameters
        ----------
        logdir : str
            Path to folder with logs.
        criterion : torch criterion

        Returns
        -------
        callbacks : list of catalyst callbacks
        """
        callbacks = self._init_base_callbacks()
#         callbacks.append(self.init_logging_callback(logdir))
        return callbacks

    def _unpack_checkpoint(self, model, criterion, optimizer):
        if self.config.CHECKPOINT_PATH is not None:
            checkpoint = load_checkpoint(self.config.CHECKPOINT_PATH)
            model = model.cuda()
            unpack_checkpoint(
                checkpoint,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
            )
        return model

    def init_all(self):
        """
        Main model initialize function.

        Returns
        -------
        training_parameters : dict
            All model parameters for catalyst dl.SupervisedRunner
        """
        datasets, loaders = self.init_dataset_and_loaders()
        model = self.init_model()
        criterion = self.init_criterion()
        optimizer, scheduler = self.init_optimizer_and_scheduler(model, loaders)
        model = self._unpack_checkpoint(model, criterion, optimizer)
        model = model.cpu()

        logdir = self.reset_logdir()
        callbacks = self.init_callbacks(logdir, criterion=criterion)

        training_parameters = {
            'datasets': datasets,
            'loaders': loaders,
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'callbacks': callbacks,
            'logdir': logdir,
        }

        return training_parameters

    def init_model(self):
        raise NotImplementedError

    def init_logging_callback(self, logdir):
        raise NotImplementedError
