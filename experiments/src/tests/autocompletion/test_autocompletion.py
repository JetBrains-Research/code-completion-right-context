import pytest

from src.preprocessing.preprocessing import LexerBasedPreprocessor
from src.modeling.gpt2 import GPT2Model

from src.generation.autocompletion import AutocompletionModel
from src.generation.generation_utils import TokenScoresPostprocessor, NextTokenChooser
from src.tests.test_tokenization import construct_spm_tokenizer

from .abstract_autocompletion_test import AbstractAutocompletionTest


class TestAutocompletion(AbstractAutocompletionTest):

    @staticmethod
    def create_random_model() -> AutocompletionModel:
        preprocesser = LexerBasedPreprocessor()
        tokenizer = construct_spm_tokenizer()
        model = GPT2Model(
            vocab_size=58, sequence_length=16, head_size=8, n_layers=2, n_heads=4,
            is_raw_output=False,
        )
        score_postprocesser = TokenScoresPostprocessor(temperature=1)
        next_token_chooser = NextTokenChooser(do_sample=False)

        return AutocompletionModel(
            preprocessor=preprocesser,
            tokenizer=tokenizer,
            model=model,
            score_postprocesser=score_postprocesser,
            next_token_chooser=next_token_chooser,
            num_beams=5,
            max_num_sequence_return=5,
        )

    @pytest.fixture(scope='session')
    def autocompletion_model(self) -> AutocompletionModel:
        return self.create_random_model()

    @pytest.mark.filterwarnings("ignore:word .*? doesn't satisfy the prefix:UserWarning")
    def test_autocompletion_with_prefix_is_working(self, autocompletion_model):
        super().test_autocompletion_with_prefix_is_working(autocompletion_model)
