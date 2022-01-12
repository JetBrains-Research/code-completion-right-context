from .base_config_initializer import BaseConfigInitializer
from .bi_gpt2 import BiGPTModel
from .bi_dataset import BiDatasetLoaderInitializer


class BiGPT2ConfigInitializer(BaseConfigInitializer):

    def init_model(self):
        model = BiGPTModel(
            vocab_size=self.config.VOCAB_SIZE * 1000,
            sequence_length=self.config.SEQUENCE_LENGTH,
            head_size=self.config.HEAD_SIZE,
            n_layers=self.config.N_LAYERS,
            n_heads=self.config.N_HEADS,
            dropout=self.config.DROPOUT,
        )
        return model

    def init_dataset_and_loaders(self):
        config = self.config
        initializer = BiDatasetLoaderInitializer(
            data_dir=config.DATA_DIR,
            tokenizer_name=config.TOKENIZER,
            vocab_size=config.VOCAB_SIZE,
            sequence_length=config.SEQUENCE_LENGTH,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            use_first_n_objects=config.use_first_n_objects,
            train_mode=config.DATASET_TRAIN_MODE,
            valid_mode=config.DATASET_VALID_MODE,
            **config.DATASET_ADDITIONAL_ARGUMENTS,
        )
        datasets, loaders = initializer.initialize_dataset_and_loaders()

        return datasets, loaders
