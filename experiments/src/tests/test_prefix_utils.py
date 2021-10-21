from src.generation.prefix_utils import PrefixMatcher
from src.preprocessing.tokenization import BaseTokenizerWrapper


class SomeTokenizer(BaseTokenizerWrapper):
    def __init__(self):
        self._token_to_id = {
            'i': 0,
            'j': 1,
            'ij': 2,
            'ijk': 3,
            'kk': 4,
            '<-': 5,
            'var0': 6,
        }
        self._wordpiece_prefix = '##'
        self._start_prefix = ''

    @property
    def token_to_id(self):
        return self._token_to_id

    @property
    def wordpiece_prefix(self):
        return self._wordpiece_prefix

    @property
    def start_prefix(self):
        return self._start_prefix

    def encode(self, text):
        return [self.token_to_id[x] for x in text.split()]


def test_oneletter_prefix():
    prefix_text = 'i'
    matcher = PrefixMatcher(
        tokenizer=SomeTokenizer(),
    )
    new_prefix = matcher.get_prefix_from_text(
        text=prefix_text
    )
    assert new_prefix.text == prefix_text
    assert new_prefix.ids == [[0], [2], [3]]


def test_strangeletter_prefix():
    prefix_text = 'ij'
    matcher = PrefixMatcher(tokenizer=SomeTokenizer())
    new_prefix = matcher.get_prefix_from_text(
        text=prefix_text
    )
    assert new_prefix.text == prefix_text
    assert new_prefix.ids == [[0], [2], [3]]


def test_no_less_letters_case():
    matcher = PrefixMatcher(tokenizer=SomeTokenizer())
    result = matcher.get_prefix_from_text(
        text='k',
        previous_text='i <- ',
        old_to_new_variables=dict(),
    )
    assert result.ids == [[4]]


def test_old_to_new_varibale_prefix_matching():
    matcher = PrefixMatcher(tokenizer=SomeTokenizer())

    old_to_new_variable = {
        'var0': 'ijkijk',
    }

    result = matcher.get_prefix_from_text(
        text='ij',
        previous_text='i <- ',
        old_to_new_variables=old_to_new_variable,
    )
    assert result.ids == [[0], [2], [3]]
