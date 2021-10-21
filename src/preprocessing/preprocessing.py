import os
import re
from typing import List, Set, Dict, Union, Any
from pygments.token import Token
from tqdm.auto import tqdm

from . import literal_handlers
from . import handlers
from .lexer_output_handler import TokenListTransformer, BaseHandler
from .r_lexer import MySLexer as SLexer


class LexerBasedPreprocessor:
    """
    Main preprocessor class.
    """
    def __init__(
            self,
            used_handlers: List[BaseHandler] = None,
            verbose: bool = True,
            protected_names: Set[str] = None,
            document_delimiter: str = ' $$$ ',
    ):
        """
        Parameters
        ----------
        used_handlers : list of BaseHandler
            Handlers which are used for preprocessing.
            If None then default handler list is used.
        verbose : bool
        protected_names : set of str
            Set of protected_names.
            protected_names are unchangeable during the preprocessing.
            For example, protected_names will not be replaced
            by universal name in VariableNameHandler.
        document_delimiter : str
            Delimiter between different documents if preprocess_folder is called.
        """
        self.lexer = SLexer(stripall=True)
        assert len(list(self.lexer.get_tokens('a[1]'))) == 5, 'Wrong lexer version'

        if used_handlers is None:
            used_handlers = [
                handlers.handle_comment,
                handlers.handle_packages,
                handlers.handle_functions,
                handlers.handle_nested_literals,
                literal_handlers.LiteralStringHandler(replace_repeating_literals=False),
                literal_handlers.LiteralNumberHandler(replace_repeating_literals=False),
                handlers.handle_delimeters,
                handlers.VariableNameHandler(),
                handlers.NonLatinHandler(),
            ]

        self.lexer_output_postprocesser = TokenListTransformer(
            handlers=used_handlers, base_protected_names=protected_names
        )
        self.verboser = tqdm if verbose else lambda x: x
        self.document_delimiter = document_delimiter

        self._regexp_for_comments_and_blank_lines = re.compile('(^#.*\n)|(\n\s*(?=\n))')

    def iter_folder(self, folder: str):
        """
        Get iterator over files in folder.
        """
        files_train = [f'{folder}/{local_path}' for local_path in os.listdir(folder)]
        for i, filename in self.verboser(enumerate(files_train)):
            with open(filename, 'r') as f:
                try:
                    text = f.read()
                    yield text
                except UnicodeDecodeError:
                    print(f"can't decode {i}th file: {filename}")

    def gather_token_statistics_from_folder(
            self,
            folder: str,
            protected_names_ratio: float = 1e-3,
    ):
        """
        Get token statistics from folder.
        If you pass protected_names to __init__
        then you don't need to call this method.


        Parameters
        ----------
        folder : str
            Path to folder.
        protected_names_ratio : float
            If fit method is called then protected_names is constructed.
            protected_names is protected_names_ratio * token_amount top tokens.
        """
        token_to_count = dict()

        for text in self.iter_folder(folder):
            tokens = list(self.lexer.get_tokens(text))
            name_tokens = [
                token_value
                for token_type, token_value in tokens
                if token_type == Token.Name
            ]
            token_to_count.update(name_tokens)
            for token_type, token_value in tokens:
                if token_type == Token.Name:
                    token_to_count.setdefault(token_value, 0)
                    token_to_count[token_value] += 1

        sorted_pairs = sorted(token_to_count.items(), key=lambda x: -x[1])
        protected_token_amount = int(len(sorted_pairs) * protected_names_ratio)
        self.lexer_output_postprocesser.base_protected_names = {
            x[0] for x in sorted_pairs[:protected_token_amount]
        }

    def preprocess_folder(self, folder: str, out_file_path: str):
        with open(out_file_path, 'w') as f:
            for text in tqdm(self.iter_folder(folder)):
                cleaned_text = self.document_delimiter + self.preprocess_code_text(text)
                f.write(cleaned_text)

    def preprocess_code_text(
            self,
            text: str,
            reset: bool = True,
            drop_last_word: str = 'never',
            return_meta_info: bool = False,
            lines_to_keep: int = None,
    ) -> Union[str, Dict[str, Any]]:
        """

        Parameters
        ----------
        text : str
        reset : bool
            If True than all internal attributes will be reset.
            It affects variable name mapping.
        drop_last_word : str
            Possible values are 'auto', 'always', 'never'.
            'always' - drop last word
            'never' - doesnt drop last word
            'auto' - drop last word according to its value
        return_meta_info : bool
            If True then bad_words list and old_name_to_new dict will be in the output.
        lines_to_keep : int
            How lines in input to keep.
            If None then all lines will be kept.

        Returns
        -------
        result : str or dict
            If return_meta_info is False.
            Result of preprocessing without additional info.
            It is returned if return_meta_info is False.

            If return_meta_info is True.
            Result of preprocessing with all additional info.
            It is returned if return_meta_info is True.
            Keys and values are:
                'output': str, preprocessed text output
                'bad_words': list os str, stop word list
                'old_name_to_new': dict, mapping from replaced names to new ones
                'last_token': str, str of cropped token (if any is present)

        bad_words : list of str
            List of stop words.
        """
        if lines_to_keep is not None:
            filtered_text = re.sub(self._regexp_for_comments_and_blank_lines, '', text)
            text = '\n'.join(filtered_text.split('\n')[-lines_to_keep:])

            # delete additional spaces
        no_space_text = "".join(text.split(' ')).strip()
        lexer_tokens = list(self.lexer.get_tokens(no_space_text))[:-1]

        try:
            if re.search(r'\s', text[-1]) and drop_last_word == 'auto':
                drop_last_word = 'never'
        except IndexError:
            pass

        if drop_last_word == 'auto':
            lexer_last_token = lexer_tokens[-1]
            if lexer_last_token[0] in {Token.Operator, Token.Punctuation}:
                drop_last_word = 'never'
            else:
                drop_last_word = 'always'

        if drop_last_word == 'always':
            preprocessed_last_token = self.lexer_output_postprocesser.transform_sequence(
                lexer_tokens[-1:]
            )
            last_token_output = "".join(preprocessed_last_token)
            lexer_tokens = lexer_tokens[:-1]
        elif drop_last_word == 'never':
            last_token_output = None
        else:
            raise TypeError(f'Wrong value for drop_last_word variable: {drop_last_word}')

        preprocessed_tokens = self.lexer_output_postprocesser.transform_sequence(
            lexer_tokens, reset=reset,
        )
        final_output = "".join(preprocessed_tokens)

        if not return_meta_info:
            return final_output

        bad_words = []
        old_name_to_new = dict()
        for handler in self.lexer_output_postprocesser.handlers:
            if hasattr(handler, 'old_name_to_new'):
                num_replaced_variables = len(handler.old_name_to_new)
                bad_words.append(f'var{num_replaced_variables}')
                old_name_to_new.update(handler.old_name_to_new)

        return {
            'output': final_output,
            'bad_words': bad_words,
            'old_name_to_new': old_name_to_new,
            'last_token': last_token_output,
        }
