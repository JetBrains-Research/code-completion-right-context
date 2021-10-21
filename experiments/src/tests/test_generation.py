import torch
import numpy as np
from numpy.testing import assert_almost_equal
from collections import Counter
from src.generation.generation_utils import TokenScoresPostprocessor, NextTokenChooser
from torch.nn.functional import softmax


def test_bad_token_filtering():
    score_postprocesser = TokenScoresPostprocessor(
        temperature=1,
    )

    input_ids = torch.tensor([[2, 3]])
    scores = torch.tensor([[5, 10, 15, 5, 5]]).float()
    bad_word_ids = [
        [[0], [3, 2], [2, 4]]
    ]

    postprocessed_scores = score_postprocesser.postprocess_next_token_scores(
        scores=scores,
        input_ids=input_ids,
        bad_word_ids=bad_word_ids,
    )

    assert postprocessed_scores.tolist() == [[-float('inf'), 10, -float('inf'), 5, 5]]


def test_good_token_filtering():
    score_postprocesser = TokenScoresPostprocessor(
        temperature=1,
    )

    input_ids = torch.tensor([[2, 3]])
    scores = torch.tensor([[5, 10, 15, 5, 5]]).float()
    good_word_ids = [
        [[0], [3, 2], [2, 4]]
    ]
    postprocessed_scores = score_postprocesser.postprocess_next_token_scores(
        scores=scores,
        input_ids=input_ids,
        good_word_ids=good_word_ids,
    )

    assert postprocessed_scores.tolist() == [[5, -float('inf'), 15, -float('inf'), -float('inf')]]


def test_choose_tokens_with_beam_seach():
    chooser = NextTokenChooser(do_sample=False)
    tmp_tensor = torch.tensor([
        [1, 2, 2, 10],
        [7, 1, 6, 0],
        [1, 2, 1, 2],
    ]).float()
    next_token_info = chooser.get_next_token_from_scores(
        scores=tmp_tensor, num_tokens=3
    )

    assert next_token_info.sequence_ids.tolist() == [0, 1, 1]
    assert next_token_info.token_ids.tolist() == [3, 0, 2]
    assert next_token_info.scores.tolist() == [10, 7, 6]


def test_choose_tokens_with_beam_seach_and_sequence_restriction():
    chooser = NextTokenChooser(do_sample=False)
    tmp_tensor = torch.tensor([
        [1, 2, 2, 10],
        [7, 1, 6, 5],
        [1, 2, 1, 3],
    ]).float()

    next_token_info = chooser.get_next_token_from_scores(
        scores=tmp_tensor, num_tokens=4, sequence_max_samples=2,
    )

    assert next_token_info.sequence_ids.tolist() == [0, 1, 1, 2]
    assert next_token_info.token_ids.tolist() == [3, 0, 2, 3]
    assert next_token_info.scores.tolist() == [10, 7, 6, 3]


def test_choose_tokens_for_boundary_lengths():
    chooser = NextTokenChooser(do_sample=False)
    tmp_tensor = torch.tensor([
        [1, 3, 2, 10],
    ]).float()

    next_token_info = chooser.get_next_token_from_scores(
        scores=tmp_tensor, num_tokens=5, sequence_max_samples=None,
    )

    assert next_token_info.sequence_ids.tolist() == [0, 0, 0, 0]
    assert next_token_info.token_ids.tolist() == [3, 1, 2, 0]
    assert next_token_info.scores.tolist() == [10, 3, 2, 1]


def test_choose_tokens_for_boundary_lengths_with_restrictions():
    chooser = NextTokenChooser(do_sample=False)
    tmp_tensor = torch.tensor([
        [1, 3, 2, 10],
    ]).float()

    next_token_info = chooser.get_next_token_from_scores(
        scores=tmp_tensor, num_tokens=10, sequence_max_samples=5,
    )

    assert next_token_info.sequence_ids.tolist() == [0, 0, 0, 0]
    assert next_token_info.token_ids.tolist() == [3, 1, 2, 0]
    assert next_token_info.scores.tolist() == [10, 3, 2, 1]


def test_token_sampling():
    chooser = NextTokenChooser(do_sample=True)
    log_probs = torch.tensor([
        [1, np.log(2) + 1, 1]
    ])

    id_counts = Counter()
    for i in range(5000):
        next_token_info = chooser.get_next_token_from_scores(
            scores=log_probs, num_tokens=1, sequence_max_samples=None,
        )
        id_counts.update([float(next_token_info.token_ids[0])])

    probs_after_sampling = np.array([
        [id_counts[0], id_counts[1], id_counts[2]]
    ], dtype=float)
    probs_after_sampling /= probs_after_sampling.sum()

    assert abs(probs_after_sampling - softmax(log_probs, dim=-1).numpy()).sum() < 0.1


def test_sampling_mode():
    chooser = NextTokenChooser(do_sample=True, top_k=1)
    log_probs = torch.tensor([
        [np.log(1 / 3), np.log(1 / 2), np.log(1 / 6)]
    ])
    next_token_info = chooser.get_next_token_from_scores(
        scores=log_probs, num_tokens=1, sequence_max_samples=None,
    )
    assert abs(next_token_info.scores - np.log(1 / 2)) < 1e-5

