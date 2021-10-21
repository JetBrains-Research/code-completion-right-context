# Generation

## Autocompletion engine

Main class for autocompletion is `AutocompletionModel` form `autocompletion.py`.
Class is constructed from other entities: preprocesser, tokenizer, model etc.
You can replace each entity with different one with the same interface.

## Prefix matching

`PrefixMatcher` from `prefix_utils.py` is used to generate tokens when cursor is inside last token.
Idea is to roll away cursor to the start of the last token and try to generate tokens
startswith the last token.
`PrefixMatcher` uses trie structure to get matching tokens.




