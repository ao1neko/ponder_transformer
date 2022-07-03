import argparse
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

from logzero import logger
from transformers import PreTrainedTokenizer, BartTokenizer, PreTrainedTokenizerFast, BartTokenizerFast, T5Tokenizer

from .BART_digits_aware_tokenizer import DigitsAwareTransformerTokenizer



__all__ = ["BartDentakuTokenizer"]


class BartDentakuTokenizer(BartTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.name_or_path=='facebook/bart-base':
            logger.warning("This tokenizer is only supported as \"facebook/bart-base\"!")
        
        self.digit_tokenizer = DigitsAwareTransformerTokenizer(self.name_or_path)
        
    

    def _tokenize(self, text, **kwargs) -> List[str]:
        tokenized_text = [token.text for token in self.digit_tokenizer.tokenize(text)]
        tokenized_text[0] = "Ġ" + tokenized_text[0]
        return tokenized_text

class T5DentakuTokenizer(T5Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.digit_tokenizer = DigitsAwareTransformerTokenizer(self.name_or_path)
        
    

    def _tokenize(self, text, **kwargs) -> List[str]:
        tokenized_text = [token.text for token in self.digit_tokenizer.tokenize(text)]
        tokenized_text[0] = "Ġ" + tokenized_text[0]
        return tokenized_text