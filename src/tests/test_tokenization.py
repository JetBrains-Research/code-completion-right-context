from src.preprocessing.tokenization import (
    SentencepieceTokenizerWrapper
)
import sentencepiece as spm
import tempfile
from pathlib import Path
import os
from typing import IO


class ReusableNamedTemporaryFile:
    """
    On Windows you can't open tempfile.NamedTemporaryFile multiple times
    this causes tests to fail, because spm_tokenizer tries to open file
    during training. This class solves the problem.
    """
    def __init__(self, mode="wb", delete=True):
        self.mode = mode
        self.delete = delete
        self.tmp_file = None
        self.tmp_file_stream = None

    @staticmethod
    def generate_random_file_name() -> Path:
        return Path(tempfile.gettempdir()) / os.urandom(24).hex()

    def __enter__(self) -> IO[bytes]:
        self.tmp_file = self.generate_random_file_name()
        while True:
            try:
                self.tmp_file.open('x').close()
                break
            except FileExistsError:
                self.tmp_file = self.generate_random_file_name()

        self.tmp_file_stream = self.tmp_file.open(self.mode)
        return self.tmp_file_stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tmp_file_stream.close()
        if self.delete:
            self.tmp_file.unlink()


def construct_spm_tokenizer():
    with ReusableNamedTemporaryFile() as f_data, tempfile.TemporaryDirectory() as d_model:
        f_data.write(
            b'var <- num\n'
            b'var <- str\n'
            b'cls$attr <- cls$attr\n'
            b'str$attr <- str$attr\n'
            b'var$attr <- var$attr\n'
            b'num$attr <- num$attr\n'
            b'cls$attr <- f(num$attr)\n'
            b'str$attr <- f(var$attr)\n'
            b'var$attr <- f(str$attr)\n'
            b'num$attr <- f(cls$attr)\n'
            b'cls.name <- cls\n'
            b'str.name <- str\n'
            b'var.name <- var\n'
            b'num.name <- num\n'
        )
        f_data.seek(0)
        spm.SentencePieceTrainer.Train(
            f'--input={f_data.name} '
            f'--model_prefix={d_model}/tok '
            f'--vocab_size=58 '
            f'--character_coverage=1 '
            f'--model_type=bpe '
            f'--split_by_unicode_script=true '
        )
        tokenizer = SentencepieceTokenizerWrapper(f'{d_model}/tok.model')

    return tokenizer


def test_encoding_decoding_is_equal_to_input():
    tokenizer = construct_spm_tokenizer()
    initial_code = 'var$attr <- f(num)'
    encoding_result = tokenizer.encode(initial_code)
    inverse_decoding_result = tokenizer.decode(encoding_result)
    assert initial_code == inverse_decoding_result


def test_tokenizer_vocab_size():
    tokenizer = construct_spm_tokenizer()
    assert len(tokenizer.id_to_token) == 58
    assert len(tokenizer.token_to_id) == 58


def test_tokenizer_tokens():
    tokenizer = construct_spm_tokenizer()
    assert 'attr' in tokenizer.token_to_id
    assert 'name' in tokenizer.token_to_id

