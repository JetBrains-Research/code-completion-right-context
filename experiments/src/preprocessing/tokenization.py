import sentencepiece as spm
import tokenizers
from typing import List, Dict
from ..utils.technical import annotations_from_parent


class BaseTokenizerWrapper:
    def encode(
            self,
            text: str,
            add_eos: bool = True,
            add_bos: bool = False
    ) -> List[int]:
        """
        Transform text to text token indexes.

        Parameters
        ----------
        text : str
            Input text.
        add_eos : bool
            Add EOS token to the end of tokens.
        add_bos : bool
            Add BOS token to the start of tokens.

        Returns
        -------
        : list of int
            Token indexes.
        """
        raise NotImplementedError

    def decode(self, tokenized_text: List[int]) -> str:
        """
        Transform text token indexes to text.

        Parameters
        ----------
        tokenized_text : list of int
            Token indexes.

        Returns
        -------
        : str
            Text.
        """
        raise NotImplementedError

    @property
    def id_to_token(self) -> Dict[int, str]:
        """
        Mapping from token indexes to initial str tokens.
        """
        raise NotImplementedError

    @property
    def token_to_id(self) -> Dict[str, int]:
        """
        Mapping from initial str tokens to token indexes.
        """
        raise NotImplementedError

    @property
    def wordpiece_prefix(self) -> str:
        """
        Str prefix for inside-word tokens start.
        """
        raise NotImplementedError

    @property
    def start_prefix(self) -> str:
        """
        Str prefix for start-word tokens start.
        """
        raise NotImplementedError

    @property
    def eos_id(self) -> int:
        """
        End of sentence token index.
        """
        raise NotImplementedError


@annotations_from_parent
class SentencepieceTokenizerWrapper(BaseTokenizerWrapper):
    def __init__(self, tokenizer_path: str):
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(tokenizer_path)

        self._token_to_id = dict()
        self._id_to_token = dict()
        for i in range(self._tokenizer.vocab_size()):
            token = self._tokenizer.id_to_piece(i)
            self._token_to_id[token] = i
            self._id_to_token[i] = token

    def encode(self, text, add_eos=True, add_bos=False):
        tokenized_text = self._tokenizer.encode(
            text, add_eos=add_eos, add_bos=add_bos
        )
        return tokenized_text

    def decode(self, tokenized_text):
        return self._tokenizer.decode(tokenized_text)

    @property
    def id_to_token(self):
        return self._id_to_token

    @property
    def token_to_id(self):
        return self._token_to_id

    @property
    def wordpiece_prefix(self):
        return ''

    @property
    def start_prefix(self):
        return '▁'

    @property
    def eos_id(self):
        return self._tokenizer.eos_id()


@annotations_from_parent
class BertWordPieceTokenizerWrapper(BaseTokenizerWrapper):
    def __init__(self, vocab_path, lowercase=False):
        self._tokenizer = tokenizers.BertWordPieceTokenizer(
            vocab_file=vocab_path,
            lowercase=lowercase,
        )
        self._token_to_id = self._tokenizer.get_vocab()
        self._id_to_token = {
            token_id: token
            for token, token_id in self._token_to_id.items()
        }

        bos_id, eos_id = self.encode('', add_bos=True, add_eos=True)
        self._eos_id = eos_id
        self._bos_token = bos_id

    def encode(self, text, add_eos=True, add_bos=False):
        tokenized_text = self._tokenizer.encode(text).ids
        if not add_eos:
            tokenized_text = tokenized_text[:-1]
        if not add_bos:
            tokenized_text = tokenized_text[1:]
        return tokenized_text

    def decode(self, tokenized_text):
        text = self._tokenizer.decode(tokenized_text)
        postprocessed_text = clever_join("", text.split())
        return postprocessed_text

    @property
    def id_to_token(self):
        return self._id_to_token

    @property
    def token_to_id(self):
        return self._token_to_id

    @property
    def wordpiece_prefix(self):
        return '##'

    @property
    def start_prefix(self):
        return ''

    @property
    def eos_id(self):
        return self._eos_id


def clever_join(sep: str, list_of_str: List[str]):
    """
    Function to join output of decode result.
    Function doesn't allow to join together two variables

    Examples
    --------
    >>> clever_join("", ["t1", "t2"])
    "t1\nt2"
    >>> clever_join("", ["t1", 2])
    "t12"

    Parameters
    ----------
    sep : str
    list_of_str : list of str

    Returns
    -------
    : str
    """
    if not list_of_str:
        return ""

    new_list_of_str = [list_of_str[0]]
    for i, elem in enumerate(list_of_str[1:], start=1):
        if elem[0].isalpha() and list_of_str[i - 1].isalpha():
            new_list_of_str.append("\n" + elem)
        else:
            new_list_of_str.append(elem)
    join_result = sep.join(new_list_of_str)
    return join_result
