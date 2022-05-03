   def tokens_to_wordpieces(
        self,
        tokens: List[Token],
        skip_tokens: List[Token] = None,
    ) -> List[Token]:
        def split_token_to_wordpieces(text) -> List[Token]:
            wordpieces = self._tokenize(text)
            wordpieces = split_digits_into_chars(
                wordpieces, self.convert_tokens_to_ids, self._unk_id
            )
            return wordpieces

        new_tokens = []
        for token in tokens:
            if skip_tokens is not None and token in skip_tokens:
                new_tokens.append(token)
            elif len(new_tokens) > 0 and new_tokens[-1].idx_end != token.idx:
                # a bit of hack so Ġ is prepended to the first wordpiece
                # cf. https://huggingface.co/transformers/_modules/transformers/models/roberta/tokenization_roberta.html#RobertaTokenizer
                new_tokens.extend(split_token_to_wordpieces(" " + token.text))
            elif (    # (追加): 最初のトークンが数字の場合
                len(new_tokens) == 0                and convert_word_to_number(token, self.include_more_numbers) is not None
            ):
                new_tokens.extend(split_token_to_wordpieces(" " + token.text))
            else:
                new_tokens.extend(split_token_to_wordpieces(token.text))
        return new_tokens



    def _old_tokens_to_wordpieces(
        self,
        tokens: List[Token],
        skip_tokens: List[Token] = None,
    ) -> List[Token]:
        def split_token_to_wordpieces(text) -> List[Token]:
            wordpieces = self._tokenize(text)
            wordpieces = split_digits_into_chars(
                wordpieces, self.convert_tokens_to_ids, self._unk_id
            )
            return wordpieces

        new_tokens = []
        for token in tokens:
            if skip_tokens is not None and token in skip_tokens:
                new_tokens.append(token)
            elif len(new_tokens) > 0 and new_tokens[-1].idx_end != token.idx:
                # a bit of hack so Ġ is prepended to the first wordpiece
                # cf. https://huggingface.co/transformers/_modules/transformers/models/roberta/tokenization_roberta.html#RobertaTokenizer
                new_tokens.extend(split_token_to_wordpieces(" " + token.text))
            else:
                new_tokens.extend(split_token_to_wordpieces(token.text))
        return new_tokens
