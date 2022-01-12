import os
import json

import joblib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



class LanguageModelDataset(Dataset):
    def __init__(
            self,
            text=None,
            text_list=None,
            reshuffle=False,
            sequence_length=512,
            batch_first=False,
            use_first_n_objects=None,
            **kwargs
    ):
        """

        Parameters
        ----------
        text : numpy ndarray
            Array of token indexes.
        text_list : list with list of int
            Each element is one document token indexes.
        reshuffle : bool
            If True then shuffle all data after each epoch.
            Works only if text_list is specified.
        sequence_length : int
            Sequence length.
        batch_first : bool
        use_first_n_objects : int or None
        """
        super(LanguageModelDataset, self).__init__()
        if text is not None and text_list is not None:
            raise TypeError('only one of the arguments text and text_list must be specifed')
        if text is None and text_list is None:
            raise TypeError('one of the arguments text and text_list must be specifed')

        self.sequence_length = sequence_length
        self.batch_first = batch_first
        self.reshuffle = reshuffle
        if text is None:
            if use_first_n_objects is not None:
                text_list = text_list[:use_first_n_objects]
            self.text_list = text_list
            self._reset_text()
        else:
            if use_first_n_objects is not None:
                text = text[:sequence_length * use_first_n_objects]
            self.text = text

    def _reset_text(self):
        token_indexes = np.random.permutation(len(self.text_list))
        text = []
        for i in token_indexes:
            text.extend(self.text_list[i])

        self.text = text
        self._getitem_counter = 0

    def __getitem__(self, i):
        if self.reshuffle:
            if self._getitem_counter >= len(self):
                self._reset_text()
            self._getitem_counter += 1

        i = i * self.sequence_length
        batch_text = self.text[i:i + self.sequence_length]
        batch_target = self.text[i + 1:i + 1 + self.sequence_length]
        batch_text = torch.tensor(batch_text).long()
        batch_target = torch.tensor(batch_target).long()

        if self.batch_first:
            batch_text = batch_text.t().contiguous()
            batch_target = batch_target.t().contiguous()

        return {'features': batch_text, 'targets': batch_target}

    def __len__(self):
        # always drop the last to get equal length sequences
        return len(self.text) // self.sequence_length - 1


class LanguageModelChunkDataset(Dataset):
    """
    Dataset for big data.
    Each context size (for example 512) sequence
    is stored in a separate file.
    """
    def __init__(self, folder_with_chunks, use_first_n_objects=None):
        self.folder_with_chunks = folder_with_chunks
        if use_first_n_objects is None:
            self._length = len(os.listdir(folder_with_chunks))
        else:
            self._length = min(len(os.listdir(folder_with_chunks)), use_first_n_objects)

    def set_length(self, length):
        self._length = length

    def __getitem__(self, i):
        with open(f'{self.folder_with_chunks}/{i}.json', 'r') as f:
            chunk_tokens, last_token = json.load(f)

        input_tokens = torch.tensor(chunk_tokens).long()
        output_tokens = torch.tensor(chunk_tokens[1:] + [last_token]).long()

        return {'features': input_tokens, 'targets': output_tokens}

    def __len__(self):
        return self._length


class DatasetLoaderInitializer:
    def __init__(
            self,
            data_dir, tokenizer_name, vocab_size,
            sequence_length, batch_size, num_workers,
            use_first_n_objects=None,
            train_mode='padding', valid_mode='lm',
            shuffle_dataset='auto',
            **kwargs
    ):
        """

        Parameters
        ----------
        data_dir : str
        tokenizer_name : str
        vocab_size : int
        sequence_length : int
        batch_size : int
        use_first_n_objects : int or None
        train_mode : str
            Possible values are 'lm' or 'padding'.
            If mode is 'padding' then each dataset object is one initial object.
            If mode is 'lm' then all objects are linked together and
            each object is the long sequence slice.
        valid_mode : str
            Same as train_mode but for valid data.
        shuffle_dataset : str
            'auto', 'never' or 'always'
        """
        for mode in [train_mode, valid_mode]:
            assert mode in {'lm', 'padding', 'chunks'}

        self.data_dir = data_dir
        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_first_n_objects = use_first_n_objects
        self.data_type_to_mode = {
            'train': train_mode,
            'valid': valid_mode
        }
        self._padding_fn = PaddingCollateFn(max_length=self.sequence_length, batch_first=True)
        self.shuffle_dataset = shuffle_dataset

    def _initialize_datasets_and_loaders_as_chunks(self, data_type):
        folder_name = f'{self.data_dir}/{data_type}_{self.tokenizer_name}{self.vocab_size}'        
        dataset = LanguageModelChunkDataset(
            folder_with_chunks=folder_name,
            use_first_n_objects=self.use_first_n_objects,
        )
        
        shuffle = True if data_type == 'train' else False
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        return dataset, loader

    def _initialize_datasets_and_loaders_as_lm(self, data_type):
        # numpy array of indexes
        data = joblib.load(f'{self.data_dir}/{data_type}_{self.tokenizer_name}{self.vocab_size}')

        if self.shuffle_dataset == 'auto':
            shuffle_dataset = True if data_type == 'train' else False
        elif self.shuffle_dataset == 'never':
            shuffle_dataset = False
        elif self.shuffle_dataset == 'always':
            shuffle_dataset = True
        else:
            raise TypeError('unknown shuffle_dataset value')
        dataset = LanguageModelDataset(
            text_list=data,
            sequence_length=self.sequence_length,
            reshuffle=shuffle_dataset,
            batch_first=True,
            use_first_n_objects=self.use_first_n_objects,
        )

        shuffle = True if data_type == 'train' else False
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )
        return dataset, loader

    def _initialize_datasets_and_loaders_with_padding(self, data_type):
        # numpy array of indexes
        # data is dataset in this case
        data = joblib.load(f'{self.data_dir}/{data_type}_{self.tokenizer_name}{self.vocab_size}')

        if self.use_first_n_objects:
            data = data[:self.use_first_n_objects]

        shuffle = True if data_type == 'train' else False
        loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=1,
            collate_fn=self._padding_fn
        )

        return data, loader

    def initialize_dataset_and_loaders(self):
        datasets = dict()
        loaders = dict()

        for data_type, mode in self.data_type_to_mode.items():
            if mode == 'lm':
                dataset, loader = self._initialize_datasets_and_loaders_as_lm(data_type)
            elif mode == 'padding':
                dataset, loader = self._initialize_datasets_and_loaders_with_padding(data_type)
            elif mode == 'chunks':
                dataset, loader = self._initialize_datasets_and_loaders_as_chunks(data_type)
            else:
                raise TypeError(f'Wrong mode: {mode}')
            datasets[data_type] = dataset
            loaders[data_type] = loader

        return datasets, loaders


class PaddingCollateFn:
    def __init__(self, max_length, batch_first=True):
        self.max_length = max_length
        self.batch_first = batch_first

    def __call__(self, sequences):
        feature_sequences = []
        target_sequences = []
        for one_sequence in sequences:
            if len(one_sequence) > self.max_length:
                feature_sequence = one_sequence[:self.max_length]
                target_sequence = one_sequence[1:self.max_length + 1]
            else:
                feature_sequence = one_sequence[:- 1] + [0] * (self.max_length - len(one_sequence) + 1)
                target_sequence = one_sequence[1:] + [0] * (self.max_length - len(one_sequence) + 1)
            feature_sequences.append(feature_sequence)
            target_sequences.append(target_sequence)

        feature_sequences = torch.tensor(feature_sequences)
        target_sequences = torch.tensor(target_sequences)

        result = {
            'features': feature_sequences,
            'targets': target_sequences,
        }
        return result
