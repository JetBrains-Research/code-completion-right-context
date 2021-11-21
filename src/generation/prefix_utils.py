import re
from pygtrie import Trie


# typing
from typing import List, Dict, Union
from ..preprocessing.tokenization import BaseTokenizerWrapper


class Prefix:
    """
    Class for text and corresponding tokens stored.
    """
    def __init__(self, text: str, ids: List[List[int]]):
        self.text = text
        self.ids = ids

    def __str__(self):
        first_ids = " ,".join([str(x) for x in self.ids[:3]])
        return f'Prefix(text={self.text}, ids=[{first_ids} + ...])'

    def __repr__(self):
        return f'Prefix(text={self.text}, ids=[{self.ids}])'


class PrefixMatcher:
    def __init__(self, tokenizer: BaseTokenizerWrapper):
        token_to_id = dict(tokenizer.token_to_id)
        # don't want to generate special symbols
        for symbol in ['#', tokenizer.wordpiece_prefix, tokenizer.start_prefix]:
            if symbol and symbol in token_to_id:
                token_to_id.pop(symbol)
        self.token_trie = Trie(token_to_id)
        self._tokenizer = tokenizer

        start_token_ids = []
        for token, token_id in token_to_id.items():
            if self._tokenizer.start_prefix:
                if token.startswith(self._tokenizer.start_prefix):
                    start_token_ids.append(token_id)
            elif self._tokenizer.wordpiece_prefix:
                if token.startswith(self._tokenizer.wordpiece_prefix):
                    start_token_ids.append(token_id)
            else:
                raise TypeError('does not understand tokenizer')
        self._start_token_ids = start_token_ids

    def _get_ids_from_trie(self, prefix_text: str) -> List[List[int]]:
        """

        Parameters
        ----------
        prefix_text : str

        Returns
        -------
        ids : list of list of int
        """
        assert prefix_text.strip() != ""

        relevant_tokens_with_id = list(self.token_trie.prefixes(prefix_text))
        try:
            complemented_tokens_with_id = list(self.token_trie.iteritems(prefix=prefix_text))
            token_is_prefix = (
                len(relevant_tokens_with_id) > 0 and
                complemented_tokens_with_id[0][0] == relevant_tokens_with_id[-1][0]
            )
            if token_is_prefix:
                relevant_tokens_with_id += complemented_tokens_with_id[1:]
            else:
                relevant_tokens_with_id += complemented_tokens_with_id
        except KeyError:
            pass

        word_ids = [
            [x[1]]
            for x in relevant_tokens_with_id
        ]
        return word_ids

    def _get_ids_from_variable_mapping(
            self,
            prefix_text: str,
            old_to_new_variables: Dict[str, str],
            is_new_token: bool = False
    ) -> List[List[int]]:
        word_ids = []

        for old_name, new_name in old_to_new_variables.items():
            if old_name.startswith(prefix_text):
                if is_new_token:
                    token_ids = self._tokenizer.encode(new_name, add_bos=False, add_eos=True)
                else:
                    #TODO: not sure it is a good solution
                    if new_name.startswith('$'):
                        raise TypeError("Name after replacement doesn't have to start with $!")
                    token_ids = self._tokenizer.encode(f'${new_name}', add_bos=False, add_eos=True)
                    token_ids = token_ids[1:]
                word_ids.append(token_ids)

        return word_ids

    def get_prefix_for_new_token(
            self,
            prefix: Union[Prefix, None],
            new_token_id: int,
            old_to_new_variables: Dict[str, str]
    ) -> Union[Prefix, None]:
        """
        Get new prefix taking into account new_token.

        Parameters
        ----------
        prefix : Prefix
        new_token_id : int
        old_to_new_variables : dict

        Returns
        -------
        new_prefix : Prefix
        """
        if prefix is None:
            return None

        new_ids_if_complex_prefix = []
        for id_list in prefix.ids:
            if id_list[0] == new_token_id and len(id_list) > 1:
                new_ids_if_complex_prefix.append(id_list[1:])
        if new_ids_if_complex_prefix:
            # just continue complex prefixes until their end
            return Prefix(text="", ids=new_ids_if_complex_prefix)

        new_token = self._tokenizer.id_to_token[new_token_id]
        if len(new_token) >= len(prefix.text):
            if new_token[:len(prefix.text)] != prefix.text:
                print(new_token[:len(prefix.text)], prefix.text)
                raise TypeError("Token doesn't satisfy prefix")
            return None

        new_text_part = prefix.text[len(new_token):]
        if self._tokenizer.wordpiece_prefix + new_text_part[:1] in self._tokenizer.token_to_id:
            new_prefix_text = self._tokenizer.wordpiece_prefix + new_text_part
        elif new_text_part[0] in self._tokenizer.token_to_id:
            new_prefix_text = new_text_part
        else:
            raise TypeError(f'Unkwnon prefix: {new_text_part}')

        new_prefix = self.get_prefix_from_text(
            new_prefix_text,
            old_to_new_variables=old_to_new_variables,
        )
        return new_prefix

    def get_prefix_from_text(
            self,
            text: str,
            previous_text: str = None,
            old_to_new_variables: Dict[str, str] = None
    ) -> Prefix:
        """
        Construct prefix class from text.

        Parameters
        ----------
        text : str
        previous_text : str
            Need to understand if we inside token or not.
        old_to_new_variables : dict
            Changed variable.

        Returns
        -------
        : Prefix
        """
        if previous_text is None:
            is_new_text = False
        else:
            is_new_text = (
                True if re.search(r'\s', previous_text[-1])
                else False
            )

        ids = self._get_ids_from_trie(
            prefix_text=text,
        )
        if is_new_text and self._tokenizer.start_prefix:
            text_with_special_char = self._tokenizer.start_prefix + text
            ids += self._get_ids_from_trie(
                prefix_text=text_with_special_char,
            )

        if old_to_new_variables is not None:
            ids_from_mapping = self._get_ids_from_variable_mapping(
                prefix_text=text,
                old_to_new_variables=old_to_new_variables,
                is_new_token=is_new_text,
            )
            ids += ids_from_mapping

        return Prefix(text=text, ids=ids)
