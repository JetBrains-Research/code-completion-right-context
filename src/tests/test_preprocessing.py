import pytest

from src.preprocessing.r_lexer import MySLexer
from src.preprocessing.lexer_output_handler import TokenListTransformer
from src.preprocessing.handlers import (
    VariableNameHandler, handle_functions, get_nested_element_tokens
)
from src.preprocessing.preprocessing import LexerBasedPreprocessor
from src.preprocessing.handlers import check_token_is_setted_by_user


def test_variable_replacing_works_two_times():
    data_first = 'x <- 1\n y <- 2'
    data_second = 'z <- 3\n x <- 4'

    lexer = MySLexer()

    transformer = TokenListTransformer(handlers=[VariableNameHandler()], base_protected_names=None)

    lexer_tokens_first = list(lexer.get_tokens(data_first))
    lexer_tokens_second = list(lexer.get_tokens(data_second))
    result_first = "".join(transformer.transform_sequence(lexer_tokens_first))
    result_second = "".join(transformer.transform_sequence(lexer_tokens_second))

    answer_first = 'var0 <- 1\n var1 <- 2\n'
    answer_second = 'var0 <- 3\n var1 <- 4\n'

    assert result_first == answer_first
    assert result_second == answer_second


def test_variable_replacing_affect_only_user_vars():
    data = 'x <- Z\n y <- 2'
    lexer = MySLexer()
    transformer = TokenListTransformer(handlers=[VariableNameHandler()], base_protected_names=None)
    lexer_tokens = list(lexer.get_tokens(data))
    result = "".join(transformer.transform_sequence(lexer_tokens))
    assert result == 'var0 <- Z\n var1 <- 2\n'


def test_variable_replacing_affect_only_user_vars_hard():
    data = 'x <- Z\n y <- x'
    lexer = MySLexer()
    transformer = TokenListTransformer(handlers=[VariableNameHandler()], base_protected_names=None)
    lexer_tokens = list(lexer.get_tokens(data))
    result = "".join(transformer.transform_sequence(lexer_tokens))
    assert result == 'var0 <- Z\n var1 <- var0\n'


def test_variable_replacing_at_the_end():
    data = 'x <- y'
    lexer = MySLexer()
    transformer = TokenListTransformer(handlers=[VariableNameHandler()], base_protected_names=None)
    lexer_tokens = list(lexer.get_tokens(data))
    result = "".join(transformer.transform_sequence(lexer_tokens))
    assert result == 'var0 <- y\n'


def test_variable_replacing_with_function_arguments():
    data = 'x <- f(t=1)\n y <- 2'
    lexer = MySLexer()
    transformer = TokenListTransformer(
        handlers=[handle_functions, VariableNameHandler()],
        base_protected_names=None
    )
    lexer_tokens = list(lexer.get_tokens(data))
    result = "".join(transformer.transform_sequence(lexer_tokens))
    assert result == 'var0 <- f(t=1)\n var1 <- 2\n'


def test_variable_replacing_with_function_arguments_hard():
    data = "x <- f(neo, zeon=2, smith=3)\n"
    lexer = MySLexer()
    transformer = TokenListTransformer(
        handlers=[handle_functions, VariableNameHandler()],
        base_protected_names=None
    )
    lexer_tokens = list(lexer.get_tokens(data))
    print(lexer_tokens)
    result = "".join(transformer.transform_sequence(lexer_tokens))
    assert result == 'var0 <- f(neo, zeon=2, smith=3)\n'


@pytest.mark.parametrize("input_str, entity_index, output_str", [
    ("x <- f(hello)", 4, "(hello)"),
    ("x <- f(hello, (a), (a, (b, c)))", 4, "(hello, (a), (a, (b, c)))"),
])
def test_nested_elements_getter(input_str, entity_index, output_str):
    lexer = MySLexer()
    lexer_tokens = list(lexer.get_tokens(input_str))
    nested_tokens = get_nested_element_tokens(lexer_tokens, entity_index)
    result = "".join([x[1] for x in nested_tokens])
    assert result == output_str


def test_main_preprocesser_simple():
    data = 'x <- f(t=1)\n y <- 2'
    preprocesser = LexerBasedPreprocessor(
        used_handlers=[handle_functions, VariableNameHandler()],
        protected_names={'x'},
    )
    result = preprocesser.preprocess_code_text(data)
    assert result == 'x<-f(t=1)\nvar0<-2'


def test_main_preprocesser_with_meta_info():
    data = 'x <- f(t=1)\n y <- 2'
    preprocesser = LexerBasedPreprocessor(
        used_handlers=[handle_functions, VariableNameHandler()],
        protected_names={'x'},
    )
    result = preprocesser.preprocess_code_text(data, return_meta_info=True)
    expected_keys = ['output', 'bad_words', 'old_name_to_new', 'last_token']
    expected_result = {
        'output': 'x<-f(t=1)\nvar0<-2',
        'bad_words': ['var1'],
        'old_name_to_new': {'y': 'var0'},
        'last_token': None,
    }
    for key in expected_keys:
        assert key in result.keys()
        assert expected_result[key] == result[key]


def test_main_preprocesser_drop_token():
    data = 'x <- f(t=1)\n y <- 2'
    preprocesser = LexerBasedPreprocessor(
        used_handlers=[handle_functions, VariableNameHandler()],
        protected_names={'x'},
    )
    result = preprocesser.preprocess_code_text(data, return_meta_info=True, drop_last_word='always')
    expected_keys = ['output', 'bad_words', 'old_name_to_new', 'last_token']
    expected_result = {
        'output': 'x<-f(t=1)\nvar0<-',
        'bad_words': ['var1'],
        'old_name_to_new': {'y': 'var0'},
        'last_token': '2',
    }
    for key in expected_keys:
        assert key in result.keys()
        assert expected_result[key] == result[key], f'wrong key: {key}'


def test_assign_token_checking():
    data = 'x <- f(t=1)\n y'
    lexer_tokens = list(MySLexer().get_tokens(data))[:-1]
    is_setted = check_token_is_setted_by_user(len(lexer_tokens) - 1, lexer_tokens)
    assert is_setted is False
