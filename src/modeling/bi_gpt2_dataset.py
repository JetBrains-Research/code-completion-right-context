import os
import json

from random import choice

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BiGPTDataset(Dataset):
    def __init__(
            self,
            text=None,
            text_list=None,
            reshuffle=False,
            sequence_length=512,
            batch_first=False,
            use_first_n_objects=None,
            right_to_left_model_shifts=[2],
    ):
        """

        Parameters
        ----------
        text : numpy ndarray
            Array of token indexes.
        text_list : list of list of int
            Each element is one document token indexes.
        reshuffle : bool
            If True then shuffle all data after each epoch.
            Works only if text_list is specified.
        sequence_length : int
            Sequence length.
        batch_first : bool
        use_first_n_objects : int or None
        right_to_left_model_shift : list, default = [2]
            Shift of the right_to_left model.
        """
        super(BiGPTDataset, self).__init__()
        if text is not None and text_list is not None:
            raise TypeError('only one of the arguments text and text_list must be specifed')
        if text is None and text_list is None:
            raise TypeError('one of the arguments text and text_list must be specifed')

        if len(right_to_left_model_shifts) < 0 or any(x < 2 for x in right_to_left_model_shifts):
            raise TypeError(f'All values in right_to_left_model_shift must be greater than 2. You give {right_to_left_model_shifts}')

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
        self.right_to_left_model_shifts = right_to_left_model_shifts

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
            
        # get random shift from sequence
        random_shift = choice(self.right_to_left_model_shifts)
        
        left_to_right_first_index = i * self.sequence_length
        left_to_right_last_index = left_to_right_first_index + self.sequence_length
        left_to_right_text = self.text[left_to_right_first_index:left_to_right_last_index]

        right_to_left_first_index = i * self.sequence_length + random_shift
        right_to_left_last_index = right_to_left_first_index + self.sequence_length
        right_to_left_text = self.text[right_to_left_first_index:right_to_left_last_index]

        target_first_index = left_to_right_first_index + 1
        target_last_index = left_to_right_last_index + 1
        target_sequence = self.text[target_first_index:target_last_index]

        left_to_right_tensor = torch.tensor(left_to_right_text).long()
        right_to_left_tensor = torch.tensor(right_to_left_text).long()
        target_tensor = torch.tensor(target_sequence).long()

        return {
            'input_tensor': left_to_right_tensor,
            'reverted_input_tensor': right_to_left_tensor,
            'targets': target_tensor,
        }

    def __len__(self):
        # always drop the last to get equal length sequences
        return len(self.text) // self.sequence_length - 1