# for typing
# to annotate classes with not yet declared class
from __future__ import annotations
from typing import List, Tuple, Set
from pygments.token import _TokenType

from ..utils.technical import annotations_from_parent


class TokenListTransformer:
    """
    Preprocess lexer tokens sequence.
    """

    def __init__(
            self,
            handlers: List[BaseHandler],
            base_protected_names: Set[str] = None,
    ):
        self.handlers = handlers
        self.base_protected_names = (
            set() if base_protected_names is None
            else set(base_protected_names)
        )

        # will have values after self.reset
        self._lexer_tokens = None
        self._output_values = None
        self._handled_elements = None
        self._protected_names = None

        self.reset()

    def reset(self):
        self._protected_names = set(self.base_protected_names)
        for handler in self.handlers:
            handler.reset()

    @property
    def lexer_tokens(self) -> List[Tuple[_TokenType, str]]:
        """
        Current code tokens.
        """
        return self._lexer_tokens

    @property
    def output_values(self) -> List[str]:
        """
        Current output tokens (only values).
        """
        return self._output_values

    @property
    def handled_elements(self) -> List[bool]:
        """
        Indicator list.
        If ith element is True then
        the ith element of lexer_tokens is already processed.
        """
        return self._handled_elements

    @property
    def protected_names(self) -> Set[str]:
        """
        Set of token values that not be transformed.
        """
        return self._protected_names

    def transform_sequence(
            self,
            lexer_tokens: List[Tuple[_TokenType, str]],
            reset: bool = True
    ) -> List[str]:
        """
        Preprocess sequence of tokens.
        Result is sequence of output tokens (only values).

        Parameters
        ----------
        lexer_tokens : list of tokens
        reset : bool
            If True then state of preprocessor will be reset.
            Set it to False if you transform different part of one sequence.

        Returns
        -------
        : list of str
            Output sequence.
        """
        self._lexer_tokens = lexer_tokens
        self._output_values = [""] * len(self._lexer_tokens)
        self._handled_elements = [False] * len(self._lexer_tokens)

        if reset:
            self.reset()

        token_iterator = enumerate(zip(self._lexer_tokens, self._handled_elements))
        for current_index, (token, already_handled) in token_iterator:
            token_type, token_value = token
            if already_handled:
                continue
            if token_value in self.protected_names:
                self._output_values[current_index] = token_value
                self._handled_elements[current_index] = True
                continue

            for one_handler in self.handlers:
                is_handled = one_handler(self, token_type, token_value, current_index)
                if is_handled:
                    break
            else:
                self._output_values[current_index] = token_value

        unprocessed_token_amount = len([elem for elem in self._handled_elements if elem is None])
        assert unprocessed_token_amount == 0, "Some tokens aren't processed"

        final_output_sequence: List[str] = []
        for i, elem in enumerate(self._output_values):
            if not elem:
                continue
            if elem == '\n':
                if len(final_output_sequence) > 0 and final_output_sequence[-1] != '\n':
                    final_output_sequence.append(elem)
            else:
                final_output_sequence.append(elem)

        return final_output_sequence


class BaseHandler:
    """
    Base abstract class to construct handlers.
    """
    def reset(self):
        """
        Reset handler to its initial state.
        """
        pass

    def __call__(
            self,
            transformer: TokenListTransformer,
            token_type: _TokenType,
            token_value: str,
            current_index: int
    ):
        """
        Handler preprocessing step.
        If handler is fit for current token then it can change transformer attributes
        (lexer_tokens, output_values, handled_elements, protected_names).

        Parameters
        ----------
        transformer : TokenListTransformer instance
        token_type : Token inherited
            Current token pygments type.
        token_value : str
            Current token value.
        current_index : int
            Current token index.

        Returns
        -------
        : bool
            If False then handler is not fit for current token processing.
            If True then token processing is considered as success.
        """
        raise NotImplementedError


def handler(function):
    """
    Decorator to make handler from function.
    """
    @annotations_from_parent
    class FunctionBasedHandler(BaseHandler):
        def __call__(self, transformer, token_type, token_value, current_index):
            return function(transformer, token_type, token_value, current_index)

        def __repr__(self):
            return f"FunctionBasedHandler based on a {str(function)} function"

    function_based_handler = FunctionBasedHandler()
    return function_based_handler
