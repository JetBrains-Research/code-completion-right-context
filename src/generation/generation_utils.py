import torch
from transformers import top_k_top_p_filtering
from torch.nn.functional import softmax

# typing
from typing import List, Union


class NextTokenInfo:
    def __init__(self, sequence_ids: torch.Tensor, token_ids: torch.Tensor, scores: torch.Tensor):
        self.sequence_ids = sequence_ids
        self.token_ids = token_ids
        self.scores = scores

    def itertriplets(self):
        return zip(self.sequence_ids, self.token_ids, self.scores)


class TokenScoresPostprocessor:
    def __init__(self, temperature: float = 1.0, penalty_theta: float = 1.0):
        self.temperature = temperature
        self.penalty_theta = penalty_theta

    @staticmethod
    def _tokens_match(
            previous_tokens: List[int],
            tokens: List[int],
            bad_token_mode: bool = True
    ) -> bool:
        if len(tokens) == 1:
            # if word tokens is just one token get it
            return True
        if len(tokens) - 1 > len(previous_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            if bad_token_mode:
                return False
        if previous_tokens[-len(tokens) + 1:] == tokens[:-1]:
            # if tokens match
            return True
        else:
            return False

    def _calc_list_compatible_ids(
            self,
            sequence_ids: torch.Tensor,
            list_to_match: List[List[int]],
    ) -> List[int]:
        sequence_ids_as_list = sequence_ids.tolist()
        compatible_ids = [
            one_word_ids[-1]
            for one_word_ids in list_to_match
            if self._tokens_match(sequence_ids_as_list, one_word_ids)
        ]
        return compatible_ids

    def postprocess_next_token_scores(
            self,
            scores: torch.Tensor,
            input_ids: torch.Tensor,
            bad_word_ids: List[Union[None, List[List[int]]]] = None,
            good_word_ids: List[Union[None, List[List[int]]]] = None,
            initial_length: int = None,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        scores : torch.tensor (n_sequences, vocab_size)
        input_ids : torch.tensor (n_sequences, sequence_length)
        bad_word_ids : list of list of list of int
            Words to ban.
            First list has length n_sequences.
            Each second-depth list is list of banned words for each sequence.
            Each third-depth list is banned word (can contain several tokens).
            Incompatible with good_word_ids: each sequence can have only one of these restrictions.
        good_word_ids : list of list of list of int
            Words that are allowed.
            Incompatible with bad_word_ids.
            Logic of structure is the same as in the bad_word_ids.
        initial_length : int


        Returns
        -------
        scores : torch.tensor(n_sequences, vocab_size)
            Postprocessed scores.
        """
        if good_word_ids is None:
            good_word_ids = [None for _ in range(scores.shape[0])]
        if bad_word_ids is None:
            bad_word_ids = [None for _ in range(scores.shape[0])]
        if initial_length is None and self.penalty_theta != 1.0:
            raise TypeError("Can't use penalty theta without initial length")

        scores = scores.clone()
        # to make all scores negative to allow theta penalty working
        if self.penalty_theta != 1:
            scores -= scores.max()

        # Temperature (higher temperature => more likely to sample low probability tokens)
        if self.temperature != 1.0:
            scores /= self.temperature
        if self.penalty_theta != 1.0:
            generated_tokens = (
                input_ids[0][:, initial_length:] 
                if isinstance(input_ids, tuple) 
                else input_ids[:, initial_length:]
            )
            for i, sequence in enumerate(generated_tokens):
                scores[i, sequence] /= self.penalty_theta

        possible_id_set = set(range(scores.shape[1]))
        for i, sequence_word_lists in enumerate(zip(bad_word_ids, good_word_ids)):
            sequence_bad_word_ids, sequence_good_word_ids = sequence_word_lists
            # if sequence_bad_word_ids is not None and sequence_good_word_ids is not None:
            #     raise TypeError('Only one of the good_words_ids and bad_words_ids must be setted')
            if sequence_bad_word_ids is None and sequence_good_word_ids is None:
                continue

            if sequence_bad_word_ids is not None:
                if isinstance(input_ids, tuple):
                    banned_ids = self._calc_list_compatible_ids(input_ids[0][i], sequence_bad_word_ids)
                else:
                    banned_ids = self._calc_list_compatible_ids(input_ids[i], sequence_bad_word_ids)                    
                scores[i, banned_ids] = -float('inf')

            if sequence_good_word_ids is not None:
                # calculate a list of not-banned tokens according to good words
                if isinstance(input_ids, tuple):
                    protected_ids = self._calc_list_compatible_ids(input_ids[0][i], sequence_good_word_ids)
                else:
                    protected_ids = self._calc_list_compatible_ids(input_ids[i], sequence_good_word_ids)                    
                banned_ids = torch.tensor(list(
                    possible_id_set
                    .difference(protected_ids)
                ))
                scores[i, banned_ids] = -float('inf')

        return scores


class NextTokenChooser:
    def __init__(self, do_sample: bool = False, top_k: int = 50, top_p: float = 1.0):
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p

    def get_next_token_from_scores(
            self,
            scores: torch.Tensor,
            num_tokens: int = 1,
            sequence_max_samples: int = None,
    ) -> NextTokenInfo:
        """
        Choose num_tokens next tokens from n_sequences * vocab_size
        token probabilities.

        Parameters
        ----------
        scores : torch.tensor (n_sequences, vocab_size)
            Log probabilities of the next token in each sequence.
        num_tokens : int
            Amount of chosen tokens.
        sequence_max_samples : int
            Don't get > sequence_max_samples from one sequence.
            If None then there is no restriction.

        Returns
        -------
        : NextTokenInfo
            next_sequence_ids : torch.tensor (num_tokens)
                Indexes of sequences to be continued.
            next_tokens : torch.tensor (num_tokens)
                Indexes (in sequence) of the chosen tokens.
            next_scores : torch.tensor (num_tokens)
                Log probabilities of the chosen tokens.
        """
        if self.do_sample:
            next_token_info = self._get_next_token_with_sampling(
                scores=scores,
                num_tokens=num_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                sequence_max_samples=sequence_max_samples,
            )
        else:
            
            if sequence_max_samples is None:
                next_token_info = self._get_next_token_no_sampling_no_sequence_restriction(
                    scores=scores,
                    num_tokens=num_tokens,
                )
            else:
                next_token_info = self._get_next_token_no_sampling_with_sequence_restriction(
                    scores=scores,
                    num_tokens=num_tokens,
                    sequence_max_samples=sequence_max_samples,
                )
        no_inf_mask = next_token_info.scores > -float('inf')
        next_token_info_without_inf = NextTokenInfo(
            sequence_ids=next_token_info.sequence_ids[no_inf_mask],
            token_ids=next_token_info.token_ids[no_inf_mask],
            scores=next_token_info.scores[no_inf_mask],
        )
        return next_token_info_without_inf

    @staticmethod
    def _get_next_token_with_sampling(
            scores: torch.Tensor,
            num_tokens: int,
            top_k: int,
            top_p: float,
            sequence_max_samples: int = None,
    ):
        # Inplace top-p/top-k filtering
        filtered_scores = top_k_top_p_filtering(
            scores.clone().detach(),
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=1,
        )

        # Sample
        probs = softmax(filtered_scores, dim=-1)

        joined_next_token_ids = []
        joined_next_scores = []
        joined_sequence_ids = []

        if sequence_max_samples is None:
            sequence_max_samples = num_tokens

        for sequence_id, one_prob_vector in enumerate(probs):
            n_sampled_samples = min(sequence_max_samples, len(one_prob_vector))
            next_token_ids = torch.multinomial(
                one_prob_vector,
                num_samples=n_sampled_samples,
                replacement=False,
            )

            joined_next_token_ids += next_token_ids.tolist()
            joined_next_scores += scores[sequence_id][next_token_ids].tolist()
            joined_sequence_ids += [sequence_id] * len(next_token_ids)

        final_next_scores, relative_next_ids = torch.topk(
            torch.tensor(joined_next_scores),
            k=min(num_tokens, len(joined_next_scores)),
        )

        next_token_info = NextTokenInfo(
            sequence_ids=torch.tensor([
                joined_sequence_ids[i] for i in relative_next_ids
            ]).to(scores.device),
            token_ids=torch.tensor([
                joined_next_token_ids[i] for i in relative_next_ids
            ]).to(scores.device),
            scores=final_next_scores.to(scores.device),
        )

        return next_token_info

    @staticmethod
    def _get_next_token_no_sampling_no_sequence_restriction(
            scores: torch.Tensor, num_tokens: int
    ) -> NextTokenInfo:
        n_sequences, vocab_size = scores.shape
        score_view_length = n_sequences * vocab_size
        next_scores, next_token_ids = torch.topk(
            scores.view(score_view_length),
            k=min(num_tokens, score_view_length),
            largest=True,
            sorted=True
        )

        next_token_info = NextTokenInfo(
            sequence_ids=(next_token_ids // vocab_size).long(),
            token_ids=(next_token_ids % vocab_size).long(),
            scores=next_scores,
        )

        return next_token_info

    @staticmethod
    def _get_next_token_no_sampling_with_sequence_restriction(
            scores: torch.Tensor,
            num_tokens: int,
            sequence_max_samples: int = None,
    ) -> NextTokenInfo:
        real_sequence_max_samples = min(sequence_max_samples, scores.shape[1])
        each_sequence_chosen_scores, each_sequence_next_token_ids = torch.topk(
            scores,
            k=real_sequence_max_samples,
            largest=True,
            sorted=False,
        )
        chosen_scores_view = each_sequence_chosen_scores.view(-1)

        next_scores, relative_ids = torch.topk(
            chosen_scores_view,
            k=min(num_tokens, len(chosen_scores_view)),
            largest=True,
            sorted=True,
        )
        next_sequence_ids = relative_ids // real_sequence_max_samples
        next_token_relative_ids = relative_ids % real_sequence_max_samples
        next_token_ids = each_sequence_next_token_ids[next_sequence_ids, next_token_relative_ids]

        next_token_info = NextTokenInfo(
            sequence_ids=next_sequence_ids,
            token_ids=next_token_ids,
            scores=next_scores,
        )

        return next_token_info
