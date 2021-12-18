import torch
from torch.nn.functional import log_softmax
import numpy as np
import re
import warnings

from .prefix_utils import Prefix, PrefixMatcher
from .postprocessing import get_first_word
from .generation_utils import NextTokenInfo


# typing
from typing import List, Dict, Iterable, Union, Tuple
from ..preprocessing import LexerBasedPreprocessor
from ..preprocessing.tokenization import BaseTokenizerWrapper
from ..modeling.base_model import BaseModel
from .generation_utils import TokenScoresPostprocessor, NextTokenChooser


class ModelOutputState:
    """
    Model state description.
    """
    def __init__(
            self,
            ids: torch.Tensor,
            known_prefixes: List[Union[None, Prefix]],
            beam_log_probs: torch.Tensor,
            output_word_to_prob: Dict[str, float],
            past_model_weights: Iterable[torch.Tensor] = None,
    ):
        """

        Parameters
        ----------
        ids : torch.Tensor of shape (n_sequences, sequence_length)
            Current input sequences.
        known_prefixes : list of Prefix
            List of current prefixes for each sequence.
        beam_log_probs : torch.Tensor
            Each sequence log probabilities
        output_word_to_prob : dict
            Output words and their probabilities.
        past_model_weights : list of torch.Tensor
            Model previous state.
        """
        self.ids = ids
        self.known_prefixes = known_prefixes
        self.beam_log_probs = beam_log_probs
        self.past_model_weights = past_model_weights
        self.output_word_to_prob = output_word_to_prob


class AutocompletionModel:
    def __init__(
            self,
            preprocessor: LexerBasedPreprocessor,
            tokenizer: BaseTokenizerWrapper,
            model: BaseModel,
            score_postprocesser: TokenScoresPostprocessor,
            next_token_chooser: NextTokenChooser,
            max_tokens_amount: int = 10,
            num_beams: int = 1,
            max_num_sequence_return: int = 1,
            verbose: bool = False,
            input_lines_to_keep: int = None,
    ):
        """

        Parameters
        ----------
        preprocessor
            Preprocessor engine.
            To preprocess input text.
        tokenizer
            Tokenizer engine.
            To transform text to sequence of indexes.
        model
            Model engine.
            To get model scores for next token.
        score_postprocesser
            Postprocessor engine.
            To transform output model scores.
        next_token_chooser
            Chooser engine.
            To get tokens from scores.
        max_tokens_amount : int
            Max amount of tokens which will be generated.
        num_beams : int
            Beam amount in beam search algorithm.
        max_num_sequence_return : int
            Maximum amount of output tokens.
        verbose : bool
        input_lines_to_keep : int
            Keep only last input_lines_to_keep lines in data.
            If None then keep all lines.
        """
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        self.model = model
        self.score_postprocesser = score_postprocesser
        self.next_token_chooser = next_token_chooser

        self.prefix_matcher = PrefixMatcher(self.tokenizer)
        self.max_tokens_amount = max_tokens_amount
        self.num_beams = num_beams
        self.max_num_sequence_return = max_num_sequence_return
        self.input_lines_to_keep = input_lines_to_keep

        #TODO: make it better
        self._replaced_number_str = None
        self._replaced_variable_str = None

        for handler in self.preprocessor.lexer_output_postprocesser.handlers:
            if hasattr(handler, 'literal_number_replacer'):
                self._replaced_number_str = handler.literal_number_replacer
            if hasattr(handler, 'variable_prefix'):
                self._replaced_variable_str = handler.variable_prefix

        self.verbose = verbose
        if hasattr(self.model, 'double_context') and self.model.double_context:
            self.double_context = True
            self.autocomplete_input = self.autocomplete_input_bi_gpt
        else:
            self.double_context = False

    def _verbose_print(self, text: str, **print_args):
        if self.verbose:
            print(text, **print_args)

    def _preprocess_data(
            self,
            input_text: str,
            drop_last_word: str = 'auto',
            is_reversed: bool = False,
            reset: bool = True,
    ) -> Tuple[torch.Tensor, List[List[int]], Dict[str, str], Union[str, None]]:
        """
        Parameters
        ----------
        input_text : str
        drop_last_word : str
            Details in LexerBasedPreprocessor.preprocess_code_text docstring

        Returns
        -------
        ids : torch.tensor
            Token indexes
        bad_word_ids : list of list of int
            List of stop words indexes.
            Each word list can contain more than one index.
        old_name_to_new : dict
            Mapping from replaced tokens to new ones.
        last_token : str
            Last cropped token. Can be empty string.
        """
        preprocessing_result = self.preprocessor.preprocess_code_text(
            input_text,
            return_meta_info=True,
            drop_last_word=drop_last_word,
            lines_to_keep=self.input_lines_to_keep,
            reset=reset,
        )
        # TODO: maybe move to preprocesser
        if self._replaced_number_str:
            preprocessing_result['bad_words'] += [self._replaced_number_str]
            
        ids = self.tokenizer.encode(
            preprocessing_result['output'],
            add_eos=True,
            add_bos=True
        )
        if is_reversed:
            # it is trade of for inverse model seq starts with bos and end with eos
            # indexes inverse for other tokens 
            bos_token = ids[0]
            eos_token = ids[-1]
            ids = [bos_token] + ids[1:-1][::-1] + [eos_token]
        bad_word_ids = [
            self.tokenizer.encode(word, add_eos=False, add_bos=False)
            for word in preprocessing_result['bad_words']
        ]

        # don't use last token because it is EOS token
        
        start_index = max(0, len(ids) - 1 - self.model.max_context_length + self.max_tokens_amount)
        ids = (
            torch.tensor(ids[start_index:-1])
            .long()
            .view(1, -1)
            .to(self.model.device)
        )

        old_name_to_new = preprocessing_result['old_name_to_new']
        last_token = preprocessing_result['last_token']

        return ids, bad_word_ids, old_name_to_new, last_token

    def _preprocess_input_prefix(
            self,
            input_prefix_text: Union[str, None],
            input_text: str,
            old_name_to_new: Dict[str, str] = None,
    ) -> Prefix:
        """
        Check if input prefix text contains strange characters.
        """
        if input_prefix_text is None:
            return input_prefix_text

        space_char_position = re.search(r'\s', input_prefix_text)
        if space_char_position:
            truncated_prefix = input_prefix_text[:space_char_position.span()[1]]
            warn_message = (
                f"Prefix '{input_prefix_text}' contains some space characters, "
                f"it is replaced by: '{truncated_prefix}'"
            )
            warnings.warn(warn_message)
            input_prefix_text = truncated_prefix

        restored_prefix = self.tokenizer.decode(self.tokenizer.encode(input_prefix_text))
        restored_prefix = "".join(restored_prefix.split())

        if restored_prefix != input_prefix_text:
            warn_message = (
                f"Prefix '{input_prefix_text}' contains some UNK characters, "
                f"it is replaced by: '{restored_prefix}'"
            )
            warnings.warn(warn_message)
            input_prefix_text = restored_prefix

        if not input_prefix_text.strip():
            return None

        prefix = self.prefix_matcher.get_prefix_from_text(
            text=input_prefix_text,
            previous_text=input_text,
            old_to_new_variables=old_name_to_new,
        )

        return prefix

    def _get_next_token_log_probs(
            self,
            current_ids: torch.Tensor,
            past_model_weights: Union[Iterable[torch.Tensor], None],
    ) -> Tuple[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Wrapper method for getting model output.
        """
        with torch.no_grad():
            next_token_log_probs, past_model_weights = self.model.get_next_token_scores(
                input_ids=current_ids,
                past=past_model_weights,
                use_cache=True,
            )

        return next_token_log_probs, past_model_weights

    def _postprocess_next_token_log_probs(
            self,
            next_token_log_probs: torch.Tensor,
            current_ids: torch.Tensor,
            bad_word_ids: List[List[int]],
            known_prefixes: List[Union[None, Prefix]],
            initial_length: int,
    ) -> torch.Tensor:
        assert len(known_prefixes) == len(next_token_log_probs)
        if self.double_context:
            assert len(known_prefixes) == len(current_ids[0])
        else:
            assert len(known_prefixes) == len(current_ids)            

        bad_word_ids_list = []
        good_word_ids_list = []

        for prefix in known_prefixes:
            bad_word_ids_list.append(bad_word_ids)

            if prefix is None:
                good_word_ids_list.append(None)
            else:
                only_next_token_prefix_ids = [
                    word_ids[:1]
                    for word_ids in prefix.ids
                ]
                good_word_ids_list.append(only_next_token_prefix_ids)

        postprocessed_scores = self.score_postprocesser.postprocess_next_token_scores(
            scores=next_token_log_probs,
            input_ids=current_ids,
            bad_word_ids=bad_word_ids_list,
            good_word_ids=good_word_ids_list,
            initial_length=initial_length,
        )

        return postprocessed_scores

    @torch.no_grad()
    def _update_model_output_state_after_one_step(
            self,
            model_state: ModelOutputState,
            next_token_info: NextTokenInfo,
            initial_length: int,
            old_name_to_new: dict,
    ) -> Tuple[ModelOutputState, bool]:
        if len(next_token_info.token_ids) == 0:
            return model_state, True
        # get temporary version of current ids and beam probabilities      
        if isinstance(model_state.ids, tuple):
            tmp_new_ids = (
                torch.cat(
                    [
                        model_state.ids[0][next_token_info.sequence_ids],
                        next_token_info.token_ids.view(-1, 1)
                    ],
                    dim=1,
                ),
                model_state.ids[1]
            )
        else:
            tmp_new_ids = torch.cat(
                [model_state.ids[next_token_info.sequence_ids], next_token_info.token_ids.view(-1, 1)],
                dim=1,
            )
        tmp_new_beam_log_probs = next_token_info.scores
        
        tmp_new_prefixes = [
            model_state.known_prefixes[i] for i in next_token_info.sequence_ids
        ]

        # update output words with probs
        # also update prefixes (because some prefixes cause new output tokens)
        if self.double_context:
            generated_ids = tmp_new_ids[0][:, initial_length:]
        else:
            generated_ids = tmp_new_ids[:, initial_length:]
        ids_to_keep = []
        new_prefixes = []
        for i, one_sequence_ids in enumerate(generated_ids):
            output_text = (
                self._postprocess_generated_tokens(one_sequence_ids)
                .strip(self.tokenizer.wordpiece_prefix)
            )
            output_word = get_first_word(output_text, self.preprocessor.lexer)
            output_word = self.preprocessor.preprocess_code_text(
                output_word,
                reset=False
            ).strip()

            need_to_keep_id = (
                output_text == output_word and
                int(one_sequence_ids[-1]) != self.tokenizer.eos_id
            )

            if need_to_keep_id:
                one_new_prefix = self.prefix_matcher.get_prefix_for_new_token(
                    prefix=tmp_new_prefixes[i],
                    new_token_id=int(one_sequence_ids[-1]),
                    old_to_new_variables=old_name_to_new,
                )
                # if prefix is <\s> then we have to stop
                is_prefix_requires_stop = (
                    one_new_prefix is not None and
                    len(one_new_prefix.ids) == 1 and
                    len(one_new_prefix.ids[0]) == 1 and
                    one_new_prefix.ids[0][0] == self.tokenizer.eos_id
                )
                if is_prefix_requires_stop:
                    need_to_keep_id = False

            if need_to_keep_id:
                new_prefixes.append(one_new_prefix)
                ids_to_keep.append(i)
                continue

            need_to_update_model_state = (
                output_word not in model_state.output_word_to_prob or
                tmp_new_beam_log_probs[i] > model_state.output_word_to_prob[output_word]
            )
            if need_to_update_model_state:
                model_state.output_word_to_prob[output_word] = float(tmp_new_beam_log_probs[i])

        if len(ids_to_keep) == 0:
            return model_state, True

        # update again to decrease the complexity of the next iteration
        if self.double_context:
            new_ids = (tmp_new_ids[0][ids_to_keep], model_state.ids[1])
        else:
            new_ids = tmp_new_ids[ids_to_keep]

        new_beam_log_probs = tmp_new_beam_log_probs[ids_to_keep]
        next_token_info = NextTokenInfo(
            sequence_ids=next_token_info.sequence_ids[ids_to_keep],
            token_ids=next_token_info.token_ids[ids_to_keep],
            scores=next_token_info.scores[ids_to_keep],
        )

        # update model weights
        need_to_update_weights = (
            model_state.past_model_weights is not None and
            model_state.past_model_weights is not (None, None)
        )

        if need_to_update_weights:
            new_past_model_weights = []
            if self.double_context:
                model_previous_state = model_state.past_model_weights[0]
            else:
                model_previous_state = model_state.past_model_weights
            for old_layer_weights in model_previous_state:
                if isinstance(old_layer_weights, (tuple, list)):
                    new_layer_weights = [
                        old_one_layer_weights[:, next_token_info.sequence_ids]
                        for old_one_layer_weights in old_layer_weights
                    ]
                else:
                    new_layer_weights = old_layer_weights[:, next_token_info.sequence_ids]
                new_past_model_weights.append(new_layer_weights)
        else:
            new_past_model_weights = None

        new_model_state = ModelOutputState(
            ids=new_ids,
            known_prefixes=new_prefixes,
            beam_log_probs=new_beam_log_probs,
            output_word_to_prob=model_state.output_word_to_prob,
            past_model_weights=(
                (new_past_model_weights, model_state.past_model_weights[1])
                if self.double_context
                else new_past_model_weights
            )
        )

        return new_model_state, False

    def _generate_next_token_ids(
            self,
            input_ids: torch.Tensor,
            bad_word_ids: List[List[int]],
            old_name_to_new: Dict[str, str],
            known_prefix: Prefix = None
    ) -> Dict[str, float]:
        """

        Parameters
        ----------
        input_ids : torch.tensor (1, n_tokens)
        bad_word_ids : list of list of int
        old_name_to_new : dict (str to str)
        known_prefix : Prefix

        Returns
        -------
        : dict
        """
        model_state = ModelOutputState(
            ids=input_ids,
            beam_log_probs=torch.zeros(
                len(input_ids[0] if self.double_context else input_ids), device=self.model.device
            ),
            known_prefixes=[known_prefix for _ in range(
                len(input_ids[0] if self.double_context else input_ids)
            )],
            past_model_weights=(None, None) if self.double_context else None,
            output_word_to_prob=dict(),
        )
        initial_length = input_ids[0].shape[1] if self.double_context else input_ids.shape[1]
        
        for i in range(self.max_tokens_amount):
            next_token_logits, past_model_weights = self._get_next_token_log_probs(
                current_ids=model_state.ids,
                past_model_weights=model_state.past_model_weights,
            )
            model_state.past_model_weights = past_model_weights

            postprocessed_scores = self._postprocess_next_token_log_probs(
                next_token_log_probs=next_token_logits,
                current_ids=model_state.ids,
                bad_word_ids=bad_word_ids,
                known_prefixes=model_state.known_prefixes,
                initial_length=initial_length,
            )

            postprocessed_scores = (
                model_state.beam_log_probs.view(len(
                    model_state.ids[0] if self.double_context else model_state.ids
                ), 1) + log_softmax(postprocessed_scores, dim=-1)
            )
            
            next_token_info = self.next_token_chooser.get_next_token_from_scores(
                postprocessed_scores,
                num_tokens=self.num_beams * 2,
                sequence_max_samples=self.num_beams,
            )

            # update input ids
            model_state, is_end = self._update_model_output_state_after_one_step(
                model_state=model_state,
                next_token_info=next_token_info,
                initial_length=initial_length,
                old_name_to_new=old_name_to_new,
            )

            # break if all sequences cant be continued or sequences amount is enough
            if is_end or len(model_state.output_word_to_prob) > self.max_num_sequence_return:
                break

        return model_state.output_word_to_prob

    def _postprocess_generated_tokens(self, generated_token_ids: torch.Tensor) -> str:
        generated_token_str = self.tokenizer.decode(generated_token_ids.tolist())
        # bad place, better to move it into preprocessing
        generated_token_str = re.sub(self._replaced_number_str, '0', generated_token_str)
        return generated_token_str

    def _postprocess_output_list(
            self,
            words_and_probs: Iterable[Tuple[str, float]],
            preprocessed_prefix: Prefix = None,
            old_name_to_new: Dict[str, str] = None,
    ) -> List[Tuple[str, float]]:
        if old_name_to_new is None:
            new_name_to_old = dict()
        else:
            new_name_to_old = {
                value: key
                for key, value in old_name_to_new.items()
            }

        if preprocessed_prefix is not None:
            stripped_prefix = (
                preprocessed_prefix.text
                .strip(self.tokenizer.start_prefix)
                .strip(self.tokenizer.wordpiece_prefix)
            )
            prefix_length = len(stripped_prefix)

        sorted_words_and_probs = sorted(words_and_probs, key=lambda x: -x[1])

        new_sorted_words_and_probs = []
        for word, word_log_prob in sorted_words_and_probs:
            old_word = new_name_to_old.get(word, word)
            if not old_word or (old_word == word and re.search('var[0-9]', word)):
                continue

            # it can be happening if varN vere generated naturally
            is_bad_prefix = (
                preprocessed_prefix is not None and
                (old_word[:prefix_length] != stripped_prefix or len(old_word) <= prefix_length)
            )
            if is_bad_prefix:
                warnings.warn(f"word {old_word} doesn't satisfy the prefix {stripped_prefix}")
                continue

            new_sorted_words_and_probs.append((old_word, word_log_prob))

        return new_sorted_words_and_probs[:self.max_num_sequence_return]

    def autocomplete_input(
            self,
            input_text: str,
            return_probs: bool = False,
            drop_last_word: str = 'auto',
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Autocomplete input.
        Get the most probable next R tokens.

        Parameters
        ----------
        input_text : text
            Input r code text
        return_probs : bool
            If True then function returns list of tuples (token, token_prob)
        drop_last_word : str
            Strategy for last semantic token in data.
            If ''

        Returns
        -------
        : list of str or list of (str, float)
        """
        input_ids, bad_word_ids, old_name_to_new, last_token = self._preprocess_data(
            input_text=input_text,
            drop_last_word=drop_last_word,
        )

        known_prefix_text = last_token

        preprocessed_prefix = (
            self._preprocess_input_prefix(
                known_prefix_text,
                input_text,
                old_name_to_new=old_name_to_new,
            )
            if known_prefix_text is not None
            else None
        )

        if preprocessed_prefix is not None:
            need_to_add_stop_words = (
                self._replaced_variable_str and
                self._replaced_variable_str.startswith(preprocessed_prefix.text)
            )
            if need_to_add_stop_words:
                bad_words_for_replaced_vars = [
                    new_var
                    for old_var, new_var in old_name_to_new.items()
                    if not old_var.lower().startswith(preprocessed_prefix.text)
                ]
                bad_word_ids_for_replaced_vars = [
                    self.tokenizer.encode(word, add_eos=False, add_bos=False)
                    for word in bad_words_for_replaced_vars
                ]
                bad_word_ids += bad_word_ids_for_replaced_vars

        self._verbose_print(f'initial input_ids shape: {input_ids.shape}')
        self._verbose_print(f'initial prefix: {preprocessed_prefix}')
        self._verbose_print(f'old_name_to_new dict: {old_name_to_new}')

        if input_ids.shape == torch.Size([1, 1]) and preprocessed_prefix is None:
            sorted_words = [
                'library', 'knitr', 'context', 'setwd', 'rm',
                'source', 'require', 'install.packages', 'options', 'data'
            ]
            sorted_words_and_probs = list(zip(sorted_words, [0.1] * 10))
        else:
            output_word_to_prob = self._generate_next_token_ids(
                input_ids=input_ids,
                bad_word_ids=bad_word_ids,
                old_name_to_new=old_name_to_new,
                known_prefix=preprocessed_prefix,
            )

            self._verbose_print(f'before postprocess output_word_to_prob: {output_word_to_prob}')
            sorted_words_and_probs = self._postprocess_output_list(
                words_and_probs=output_word_to_prob.items(),
                preprocessed_prefix=preprocessed_prefix,
                old_name_to_new=old_name_to_new,
            )
            self._verbose_print(f'after postprocess output_word_to_prob: {sorted_words_and_probs}')

        if return_probs:
            sorted_words_and_probs = [(x[0], np.exp(x[1])) for x in sorted_words_and_probs]
            return sorted_words_and_probs
        else:
            return [x[0] for x in sorted_words_and_probs]
        
        
        
    def autocomplete_input_bi_gpt(
            self,
            input_text: tuple,
            return_probs: bool = False,
            drop_last_word: str = 'auto',
    ) -> Union[List[str], List[Tuple[str, float]]]:
        assert len(input_text) == 2
        left_text, right_text = input_text
        # left model
        (
            left_ids,
            bad_word_ids,
            left_old_name_to_new,
            left_last_token
        ) = self._preprocess_data(
            input_text=left_text,
            drop_last_word=drop_last_word,
            reset=True,
        )
        
        # right model
        (
            right_ids,
            right_bad_word_ids,
            right_old_name_to_new,
            right_last_token
        ) = self._preprocess_data(
            input_text=right_text,
            drop_last_word='never',
            is_reversed=True,
            reset=False,
        )

        # union of left_old_name_to_new and right_old_name_to_new
        old_name_to_new = left_old_name_to_new

        # left model
        left_known_prefix_text = left_last_token

        # left model        
        left_preprocessed_prefix = (
            self._preprocess_input_prefix(
                left_known_prefix_text,
                left_text,
                old_name_to_new=old_name_to_new,
            )
            if left_known_prefix_text is not None
            else None
        )

        # add loop in left amd right prefixes
        if left_preprocessed_prefix is not None:
            need_to_add_stop_words = (
                self._replaced_variable_str and
                self._replaced_variable_str.startswith(left_preprocessed_prefix.text)
            )
            if need_to_add_stop_words:
                bad_words_for_replaced_vars = [
                    new_var
                    for old_var, new_var in old_name_to_new.items()
                    if not old_var.lower().startswith(left_preprocessed_prefix.text)
                ]
                bad_word_ids_for_replaced_vars = [
                    self.tokenizer.encode(word, add_eos=False, add_bos=False)
                    for word in bad_words_for_replaced_vars
                ]
                bad_word_ids += bad_word_ids_for_replaced_vars
                    
                    
        self._verbose_print(f'initial left_ids shape: {left_ids.shape}')
        self._verbose_print(f'initial prefix: {left_preprocessed_prefix}')
        self._verbose_print(f'initial input_ids shape: {right_ids.shape}')
        self._verbose_print(f'old_name_to_new dict: {old_name_to_new}')

        if (
            left_ids.shape == torch.Size([1, 1]) and left_preprocessed_prefix is None
        ):
            sorted_words = [
                'library', 'knitr', 'context', 'setwd', 'rm',
                'source', 'require', 'install.packages', 'options', 'data'
            ]
            sorted_words_and_probs = list(zip(sorted_words, [0.1] * 10))
        else:
            output_word_to_prob = self._generate_next_token_ids(
                input_ids=(left_ids, right_ids),
                bad_word_ids=bad_word_ids,
                old_name_to_new=old_name_to_new,
                known_prefix=left_preprocessed_prefix,
            )

            self._verbose_print(f'before postprocess output_word_to_prob: {output_word_to_prob}')
            sorted_words_and_probs = self._postprocess_output_list(
                words_and_probs=output_word_to_prob.items(),
                preprocessed_prefix=left_preprocessed_prefix,
                old_name_to_new=old_name_to_new,
            )
            self._verbose_print(f'after postprocess output_word_to_prob: {sorted_words_and_probs}')

        if return_probs:
            sorted_words_and_probs = [(x[0], np.exp(x[1])) for x in sorted_words_and_probs]
            return sorted_words_and_probs
        else:
            return [x[0] for x in sorted_words_and_probs]        
