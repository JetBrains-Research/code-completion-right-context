import numpy as np
from src.modeling.bi_dataset import BiGPTDataset
import torch

def test_dataset_getitem():
    dataset = BiGPTDataset(
        text=[1, 2, 3, 4, 5, 6],
        reshuffle=False,
        sequence_length=5,
        right_to_left_model_shifts=[2],
    )
    first_element = dataset[0]
    assert first_element['input_tensor'].tolist() == [1, 2, 3, 4, 5]
    assert first_element['reverted_input_tensor'].tolist() == [6, 5, 4, 3]
    assert first_element['targets'].tolist() == [2, 3, 4, 5, 6]
