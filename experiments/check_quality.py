import json
from tqdm import tqdm

from catalyst.utils import load_checkpoint, unpack_checkpoint
import numpy as np
import pandas as pd

from run_gpt_ddp_training import GPT2ConfigInitializer
from src.preprocessing.preprocessing import LexerBasedPreprocessor
from src.preprocessing.tokenization import SentencepieceTokenizerWrapper
from src.generation.autocompletion import AutocompletionModel
from src.generation.generation_utils import TokenScoresPostprocessor, NextTokenChooser
from src.utils.metrics import reciprocal_rank, relevant_in_k


# fix it later to be able to import any config
from gpt_config import Config


class QualityChecker:
    def __init__(self, data_path, top_tokens_path, tokenizer_path, model_path, cuda_device):
        self.data_df = self.create_df(data_path)

        with open(top_tokens_path, 'r') as f_in:
            top_tokens = json.load(f_in)
        preprocesser = LexerBasedPreprocessor(protected_names=top_tokens)

        tokenizer = SentencepieceTokenizerWrapper(tokenizer_path)

        initializer = GPT2ConfigInitializer(Config)
        model = initializer.init_model()
        checkpoint = load_checkpoint(model_path)
        unpack_checkpoint(checkpoint=checkpoint, model=model)

        model = model.eval()
        model = model.to(cuda_device)

        self.autocompletion_model = AutocompletionModel(
            preprocessor=preprocesser,
            tokenizer=tokenizer,
            model=model,
            score_postprocesser=TokenScoresPostprocessor(temperature=1.5, penalty_theta=0.5),
            next_token_chooser=NextTokenChooser(do_sample=False),
            max_tokens_amount=5,
            num_beams=5,
            max_num_sequence_return=20,
            input_lines_to_keep=100,
        )

    def create_df(self, directory):
        dict_for_df = dict()
        with open(directory, 'r') as f:
            for line in f:
                d = json.loads(line)
                for key in d:
                    dict_for_df.setdefault(key, [])
                    dict_for_df[key].append(d[key])
        return pd.DataFrame(dict_for_df)

    def check_quality(self):
        model_outputs = []
        real_outputs = []
        bad_indexes = []
        for i, elem in tqdm(self.data_df.iterrows()):
            try:
                test_sample = elem['before_cursor']
                one_model_outputs = self.autocompletion_model.autocomplete_input(
                    test_sample,
                    drop_last_word='always' if elem['prefix'] == 'prefix' else 'never',
                )
                one_real_output = elem['after_cursor_token']
                model_outputs.append(one_model_outputs)
                real_outputs.append(one_real_output)
            except Exception as e:
                print(i, e)
                bad_indexes.append(i)

        real_o = real_outputs
        model_o = model_outputs
        relevances = [
            [int(x == one_r_o) for x in one_model_o]
            for one_r_o, one_model_o in zip(real_o, model_o)
        ]
        key_metrics = [
            [relevant_in_k(one_r, k=k) for k in range(1, 6)] + [reciprocal_rank(one_r)]
            if one_r else [0] * 6
            for one_r in relevances
        ]
        key_metrics = np.array(key_metrics).mean(axis=0)

        return key_metrics
