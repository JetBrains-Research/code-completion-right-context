from pygments.token import Token

# for typing
from .lexer_output_handler import BaseHandler, handler

REPLACED_NUMBER = '.NU.'
REPLACED_STRING = "''"


class LiteralNumberHandler(BaseHandler):
    def __init__(
            self,
            literal_number_replacer: str = REPLACED_NUMBER,
            replace_repeating_literals: bool = False,
    ):
        """

        Parameters
        ----------
        literal_number_replacer : str
            Placeholder value.
        replace_repeating_literals : bool
            If True then any sequence of numerical literals separated by ,
            will be transformed to {literal_number_replacer}{amount of literals}.
            For example, if literal_number_replacer is .NU. then
            "10, 1, 12" will be transformed to .NU.3.
        """
        self.literal_number_replacer = literal_number_replacer
        self.replace_repeating_literals = replace_repeating_literals

    def __call__(self, transformer, token_type, token_value, current_index):
        """
        Replace literal numbers with placeholder
        """
        if not token_type == Token.Literal.Number:
            return False

        if self.replace_repeating_literals:
            last_literal_index = current_index

            while (
                    last_literal_index + 2 < len(transformer.lexer_tokens) and
                    transformer.lexer_tokens[last_literal_index + 1][1] == ',' and
                    transformer.lexer_tokens[last_literal_index + 2][0] == Token.Literal.Number
            ):
                transformer.handled_elements[last_literal_index + 1] = True
                transformer.handled_elements[last_literal_index + 2] = True

                last_literal_index += 2

            literal_amount = 1 + (last_literal_index - current_index) // 2

            if literal_amount > 1:
                transformer.output_values[
                    current_index] = f'{self.literal_number_replacer}{literal_amount}'
                transformer.handled_elements[current_index] = True
                return True

        transformer.output_values[current_index] = self.literal_number_replacer
        transformer.handled_elements[current_index] = True
        return True


class LiteralStringHandler(BaseHandler):
    def __init__(
            self,
            literal_string_replacer: str = REPLACED_NUMBER,
            replace_repeating_literals: bool = False,
    ):
        """

        Parameters
        ----------
        literal_string_replacer : str
            Placeholder value.
        replace_repeating_literals : bool
            If True then any sequence of numerical literals separated by ,
            will be transformed to {literal_string_replacer}{amount of literals}.
            For example, if literal_number_replacer is .NU. then
            "'1', '23'" will be transformed to ''2
        """
        self.literal_number_replacer = literal_string_replacer
        self.replace_repeating_literals = replace_repeating_literals

    def __call__(self, transformer, token_type, token_value, current_index):
        """
        Replace literal numbers with placeholder
        """
        if not token_type == Token.Literal.String:
            return False

        if token_value in {'"', "'"}:
            transformer.handled_elements[current_index] = True
            return True

        if self.replace_repeating_literals:
            last_literal_index = current_index
            while (
                    last_literal_index + 3 < len(transformer.lexer_tokens) and
                    transformer.lexer_tokens[last_literal_index + 1][1] == ',' and
                    transformer.lexer_tokens[last_literal_index + 2][0] == Token.Literal.String and
                    transformer.lexer_tokens[last_literal_index + 3][0] == Token.Literal.String
            ):
                transformer.handled_elements[last_literal_index + 1] = True
                transformer.handled_elements[last_literal_index + 2] = True
                transformer.handled_elements[last_literal_index + 3] = True

                last_literal_index += 3

            literal_amount = 1 + (last_literal_index - current_index) // 3

            if literal_amount > 1:
                transformer.output_values[
                    current_index] = f'{self.literal_number_replacer}{literal_amount}'
                transformer.handled_elements[current_index] = True
                return True

        if token_value not in {'"', "'"}:
            transformer.output_values[current_index] = "''"
            transformer.handled_elements[current_index] = True

        return True
