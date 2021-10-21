from abc import ABC, abstractmethod
import pytest

from src.generation.autocompletion import AutocompletionModel


class AbstractAutocompletionTest(ABC):

    @pytest.fixture
    @abstractmethod
    def autocompletion_model(self, *args, **kwargs) -> AutocompletionModel:
        pass

    def test_autocompletion_is_working(self, autocompletion_model):
        data = 'var <- f(num)\n str.name <-'
        result = autocompletion_model.autocomplete_input(
            input_text=data,
            drop_last_word='never'
        )
        assert len(result) > 0

    def test_autocompletion_with_prefix_is_working(self, autocompletion_model):
        data = 'var <- f(num)\n str.name <- str$at'
        result = autocompletion_model.autocomplete_input(
            input_text=data,
            drop_last_word='always'
        )
        assert len(result) > 0


class AbstractTrainedAutocompletionTest(AbstractAutocompletionTest, ABC):

    @pytest.mark.trained_model_test
    def test_autocompletion_with_long_variables(self, autocompletion_model):
        test_sample = """
        LongStrangeNameForR <- 123
        max_veryLongStrangeNameForR <- max(LongStrangeNameForR)
        min_veryLongStrangeNameForR <- min(LongStrangeNameForR)
        abs_veryLongStrangeNameForR <- abs(Lo
        """.strip()
        result_with_prefix = autocompletion_model.autocomplete_input(
            test_sample,
            drop_last_word='always',
        )
        assert 'LongStrangeNameForR' in result_with_prefix
