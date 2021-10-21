from pygments.token import Token
import warnings
import unicodedata

# for typing
from .lexer_output_handler import BaseHandler, handler, TokenListTransformer
from pygments.token import _TokenType
from typing import List, Tuple

REPLACED_VARIABLE = 'var'
REPLACED_NONLATIN = '.NL.'


@handler
def handle_comment(transformer, token_type, token_value, current_index):
    """
    Delete comments
    """
    if token_type == Token.Comment.Single:
        transformer.handled_elements[current_index] = True
        return True


@handler
def handle_nested_literals(transformer, token_type, token_value, current_index):
    """
    Delete nested entities content.
    For example, c(1, 2, 3) -> c() or print('helloworld') -> print()
    """
    if token_type == Token.Name.Function and token_value in {'c', 'print', 'message'}:
        try:
            nested_element_tokens = get_nested_element_tokens(
                transformer.lexer_tokens, current_index
            )
        except StopIteration:
            warn_nested_structure_fragment(transformer, current_index)
            return False

        transformer.output_values[current_index] = f'{token_value}()'
        transformer.handled_elements[current_index] = True
        for i in range(len(nested_element_tokens)):
            transformer.handled_elements[current_index + 1 + i] = True

        return True


@handler
def handle_delimeters(transformer, token_type, token_value, current_index):
    """
    Squeeze delemeters.
    For example, "   " -> " ", "     \n" -> "\n"
    """
    if check_token_is_delimeter(token_type, token_value):
        if '\n' in token_value:
            transformer.output_values[current_index] = '\n'
        else:
            transformer.output_values[current_index] = ' '
        return True


@handler
def handle_packages(transformer, token_type, token_value, current_index):
    """
    Add package names to protected names
    """
    if token_type == Token.Name.Namespace:
        transformer.protected_names.add(token_value)
        transformer.output_values[current_index] = token_value
        transformer.handled_elements[current_index] = True
        return True


@handler
def handle_functions(transformer, token_type, token_value, current_index):
    """
    Add function arguments to protected names
    """
    if token_type == Token.Name.Function:
        try:
            nested_element_tokens = get_nested_element_tokens(transformer.lexer_tokens, current_index)
        except StopIteration:
            warn_nested_structure_fragment(transformer, current_index)
            return False

        for i, one_function_token in enumerate(nested_element_tokens):
            if check_token_is_function_argument(one_function_token, nested_element_tokens[i + 1:i + 3]):
                transformer.handled_elements[current_index + i + 1] = True
                transformer.output_values[current_index + i + 1] = one_function_token[1]

        return False


class NonLatinHandler(BaseHandler):
    """
    Transform non-latine name to latine analogue.
    """
    def __init__(self, nonlatin_replacer=REPLACED_NONLATIN):
        self.nonlatin_replacer = nonlatin_replacer

    def get_translation_variant(self, text):
        text_with_replaced_letters = [
            (
                unicodedata
                .normalize('NFKD', symbol)
                .encode('ascii', 'replace')
                .decode('utf-8')
                .strip('?')
            )
            for symbol in text
        ]

        result = "".join([
            symbol if symbol else self.nonlatin_replacer
            for symbol in text_with_replaced_letters
        ])

        return result

    def __call__(self, transformer, token_type, token_value, current_index):
        if check_token_has_subptype_of_type(token_type, Token.Name):
            new_token_name = self.get_translation_variant(token_value)
            transformer.handled_elements[current_index] = True
            transformer.output_values[current_index] = new_token_name
            return True


class VariableNameHandler(BaseHandler):
    """
    Replace rare user variables by unified names (varN).
    """
    def __init__(self, variable_prefix=REPLACED_VARIABLE):
        self.reset()
        self.variable_prefix = variable_prefix

    def reset(self):
        self.old_name_to_new = dict()
        self.variable_count = 0

    @property
    def new_name_to_old(self):
        new_name_to_old = {
            value: key
            for key, value in self.old_name_to_new.items()
        }
        return new_name_to_old

    def __call__(self, transformer, token_type, token_value, current_index):
        is_handler_case = (
            (token_type == Token.Name or token_type == Token.Name.Function) and
            token_value not in transformer.protected_names
        )
        if not is_handler_case:
            return False

        if token_value not in self.old_name_to_new:
            token_is_setted_by_user = check_token_is_setted_by_user(
                current_index, transformer.lexer_tokens
            )
            if not token_is_setted_by_user:
                return False

            self.old_name_to_new[token_value] = f'{self.variable_prefix}{self.variable_count}'
            self.variable_count += 1
        transformer.output_values[current_index] = self.old_name_to_new[token_value]
        transformer.handled_elements[current_index] = True
        return True


def check_token_is_delimeter(token_type: _TokenType, token_value: str):
    is_delimeter = (
        token_type == Token.Text and
        set(token_value).issubset({' ', '\n'})
    )
    return is_delimeter


def check_token_is_assigning_operator(token_type: _TokenType, token_value: str):
    is_assigning_operator = (
        token_type == Token.Operator and
        token_value in {'<-', '='}
    )
    return is_assigning_operator


def check_token_is_function_argument(
        main_token: Tuple[_TokenType, str],
        next_tokens: List[Tuple[_TokenType, str]],
        only_key_arguments: bool = True
):
    main_token_type, main_token_value = main_token
    if main_token_type != Token.Name:
        return False

    for one_token_type, one_token_value in next_tokens:
        if check_token_is_delimeter(one_token_type, one_token_value):
            continue

        is_argument = (
            (one_token_type == Token.Operator and one_token_value == '=') or
            (
                not only_key_arguments and
                one_token_type == Token.Punctuation and one_token_value == ','
            )
        )

        return is_argument


def check_token_is_setted_by_user(token_index: int, lexer_tokens: List[Tuple[_TokenType, str]]):
    if token_index + 1 >= len(lexer_tokens):
        return False

    next_first_token = lexer_tokens[token_index + 1]
    if check_token_is_assigning_operator(*next_first_token):
        return True

    if token_index + 2 >= len(lexer_tokens):
        return False

    next_second_token = lexer_tokens[token_index + 2]
    token_is_setted_by_user = (
        check_token_is_delimeter(*next_first_token) and
        check_token_is_assigning_operator(*next_second_token)
    )
    return token_is_setted_by_user


def get_nested_element_tokens(tokens: List[Tuple[_TokenType, str]], current_token_index: int):
    """
    Get all tokens for chosen entity that starts with '('.
    Content is all tokens between the first '(' and the last ')' inclusively.
    For example, for 'c(1, 2, 3)' tokens content is '(1, 2, 3)' tokens.

    Parameters
    ----------
    tokens : list of tuples
        List of tokens
    current_token_index : int
        First entity index.
        For example, it can be index of 'c' function.

    Returns
    -------
    : list of tuples
        List of token
    """
    content_tokens = []
    token_iterator = iter(tokens[current_token_index + 1:])

    # check that first token is '('
    token_type, token_value = next(token_iterator)
    if token_type == Token.Punctuation and token_value == '(':
        parenthesis_stack = [token_value]
        content_tokens.append((token_type, token_value))
    else:
        raise TypeError(f'first element must be "(" but it is {token_value}')

    while True:
        token_type, token_value = next(token_iterator)
        content_tokens.append((token_type, token_value))
        if token_type == Token.Punctuation:
            if token_value in {'(', ')'}:
                parenthesis_stack.append(token_value)
        if len(parenthesis_stack) >= 2:
            if parenthesis_stack[-1] == ')' and parenthesis_stack[-2] == '(':
                parenthesis_stack = parenthesis_stack[:-2]
        if len(parenthesis_stack) == 0:
            break

    return content_tokens


def warn_nested_structure_fragment(transformer: TokenListTransformer, current_index: int):
    relevant_fragment = "".join([
        x[1]
        for x in transformer.lexer_tokens[current_index:current_index + 5]
    ])
    warn_message = (
        f"Can't handle nested structure on position {current_index}, "
        f'fragment: {relevant_fragment}'
    )
    warnings.warn(warn_message)


def check_token_has_subptype_of_type(token_type: _TokenType, required_type: _TokenType):
    current_parent_type = token_type
    while True:
        if current_parent_type == required_type:
            return True
        try:
            current_parent_type = current_parent_type.parent
        except AttributeError:
            return False
