# Preprocessing

## LexerBasedPreprocesser

Main high-level preprocessing class is LexerBasedPreprocesser (from preprocessing.py). Preprocessing pipeline consists of several steps:
1. Raw code is preprocessed by lexer. Input is str, output is list of tuples (token_type, token_value). 
Lexer is implemented in r_lexer.py module (MySLexer class).
Implementation is based on a pygments Slexer.

2. Lexer output list is transformed to a list of preprocessed strings.
Implementation details are given below.

3. Previous step result is finally obtained in LexerBasedPreprocesser. 
All tokens are joined and some additional steps are performed.
For example, forbidden for generation word list is calculated 

## TokenListTransformer

TokenListTransformer (from lexer_output_handler.py) is list to list tool. 
TokenListTransformer manipulates with two lists with the same length as input list:
1. list of output values
2. list of bool indicators (if ith element is True the ith input token is already handled)  

Transformer consist of several handlers. 
All inner preprocessing logic has to be placed inside handlers.
Transformer handles tokens sequentially.
On each step each transformer handler tries to preprocess next token.
Each handler gets transformer instance, token type, token value and current token index as input.
Handler returns True if it preprocess token successfully and returns False otherwise.
If handler returns True then transformer starts to handle next token.

Preprocessing complexity is supposed (but no guaranteed) to be linear.
To achieve linear complexity you have to set bool indicators to True for all affected tokens.

## Modules
* r_lexer — lexer class 
* preprocessing — high-level preprocessing pipeline implementation
* lexer_output_handler — second step preprocessing implementation
* handlers — basic preprocesser components implementation
* tokenization — tokenizer wrappers


## F.A.Q.

**What is the common way to change preprocessing?**

The simpliest way is to write new handler (subclass of BaseHandler) and
add it to the handler list of LexerBasedPreprocessor.
Also note that handler order is important.
