from pygments.token import Token
from ..preprocessing.r_lexer import MySLexer


END_COMPLETION_TOKEN_TYPE = {
    Token.Name,
    Token.Name.Function,
    Token.Keyword.Constant,
    Token.Literal.Number,
    Token.Operator,
    Token.Punctuation,
}


def get_first_word(code_string: str, lexer: MySLexer) -> str:
    """
    Get first word from str code.

    Parameters
    ----------
    code_string : str
    lexer : MySLexer instance

    Returns
    -------
    : str
    """
    tokens = list(lexer.get_tokens(code_string))
    related_tokens = []
    for one_token in tokens:
        if one_token == (Token.Punctuation, ';'):
            break
        related_tokens.append(one_token[1])
        # to prevent ')token' in the output
        if one_token[0] in END_COMPLETION_TOKEN_TYPE:
            break
    first_word = "".join(related_tokens).strip()
    return first_word
