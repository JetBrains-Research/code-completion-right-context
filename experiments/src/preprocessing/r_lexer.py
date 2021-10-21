from pygments.lexers.r import SLexer

from pygments.lexer import Lexer, RegexLexer, include, do_insertions, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Generic


class MySLexer(SLexer):
    # delete . from SLexer valid_name
    valid_name = r'`[^`\\]*(?:\\.[^`\\]*)*`|(?:[a-zA-Z]|\.[A-Za-z_.])[\w.]*'
    tokens = {
        'comments': [
            (r'#.*$', Comment.Single),
        ],
        'valid_name': [
            (valid_name, Name),
        ],
        'punctuation': [
            (r'\[{1,2}|\]{1,2}|\(|\)|;|,', Punctuation),
        ],
        'keywords': [
            (r'(if|else|for|while|repeat|in|next|break|return|switch|function)'
             r'(?![\w.])',
             Keyword.Reserved),
        ],
        'packages': [
            (r'(library|require)(\()(%s)' % valid_name,
             bygroups(Name.Function, Punctuation, Name.Namespace)),
        ],
        'operators': [
            (r'<<?-|->>?|-|==|<=|>=|<|>|&&?|!=|\|\|?|\?', Operator),
            (r'\*|\+|\^|/|!|%[^%]*%|=|~|\$|@|:{1,3}', Operator),
        ],
        'builtin_symbols': [
            (r'(NULL|NA(_(integer|real|complex|character)_)?|'
             r'letters|LETTERS|Inf|TRUE|FALSE|NaN|pi|\.\.(\.|[0-9]+))'
             r'(?![\w.])',
             Keyword.Constant),
            (r'(T|F)\b', Name.Builtin.Pseudo),
        ],
        'numbers': [
            # hex number
            (r'0[xX][a-fA-F0-9]+([pP][0-9]+)?[Li]?', Number.Hex),
            # decimal number
            # added support for float numbers like N.
            (
                r'[+-]?('
                r'\.[0-9]+'
                r'|[0-9]+\.'
                r'|[0-9]+(\.[0-9]+)?'
                r'|\.'
                r')'
                r'([eE][+-]?[0-9]+)?[Li]?',
                Number
            ),
        ],
        'statements': [
            include('comments'),
            # whitespaces
            (r'\s+', Text),
            (r'\'', String, 'string_squote'),
            (r'\"', String, 'string_dquote'),
            include('builtin_symbols'),
            include('valid_name'),
            include('numbers'),
            include('keywords'),
            include('punctuation'),
            include('operators'),
        ],
        'root': [
            # calls:
            include('packages'),
            (r'(%s)\s*(?=\()' % valid_name, Name.Function),
            include('statements'),
            # blocks:
            (r'\{|\}', Punctuation),
            # (r'\{', Punctuation, 'block'),
            (r'.', Text),
        ],
        # 'block': [
        #    include('statements'),
        #    ('\{', Punctuation, '#push'),
        #    ('\}', Punctuation, '#pop')
        # ],
        'string_squote': [
            (r'([^\'\\]|\\.)*\'', String, '#pop'),
        ],
        'string_dquote': [
            (r'([^"\\]|\\.)*"', String, '#pop'),
        ],
    }
