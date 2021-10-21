import torch
from src.modeling.gpt2 import GPT2Model


def test_gpt2_basic():
    model = GPT2Model(
        vocab_size=128, sequence_length=16, head_size=8, n_layers=2, n_heads=4,
        is_raw_output=False,
    )
    input_ids = torch.tensor([[0, 10, 11, 15, 1]])
    result, past = model.get_next_token_scores(input_ids, use_cache=True)
    assert result.shape == (1, 128)
    assert len(past) == 2
    # (2, n_sequences, n_heads, sequence length, head_size / n_heads)
    assert past[0].shape == (2, 1, 4, 5, 2)


def test_gpt2_with_past():
    model = GPT2Model(
        vocab_size=128, sequence_length=16, head_size=8, n_layers=2, n_heads=4,
        is_raw_output=False,
    )
    input_ids = torch.tensor([[0, 10, 11, 15, 1]])
    after_first_result, after_first_past = model.get_next_token_scores(input_ids)
    after_first_input_ids = torch.tensor([[0, 10, 11, 15, 1, 5]])
    after_second_result, after_second_past = model.get_next_token_scores(
        after_first_input_ids,
        use_cache=True,
        past=after_first_past,
    )
    assert after_second_result.shape == (1, 128)
    assert len(after_second_past) == 2
    assert after_second_past[0].shape == (2, 1, 4, 6, 2)
