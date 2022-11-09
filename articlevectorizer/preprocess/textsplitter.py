import types
import nltk
nltk.download('punkt')
from nltk import tokenize

import spacy
from spacy.lang.en import English
from sklearn.exceptions import NotFittedError
from articlevectorizer.base import VectorizerBase

import re


class TextSplitter(VectorizerBase):

    """
    A class that takes a long string as an input,n_words to keep and a tokenizer function.
    Returns a concatinated text started with half of the words from the start and half of the words from the end.

    The choices of tokenizers currently are:
    - any spaCy English language, like `spacy.load('en_core_web_lg')`
    - NLTK `tokenize`

    The NLTK option is an order of magnitude faster, but spaCy might be more accurate in certain complicated cases.

    It will almost always return more words than required, as it does not cut the sentences if they overload above the threshold.
    The above is intentional as cutting a string is relatively easy and most models e.g. SentenceTransformers, would do that for us.

    Example:
    from nltk import tokenize
    import spacy

    text = "This is an egg. This is a chicken. This is a bird. Is the bird a chicken too?"
    text_2 = "One. Two. Three. Four. Five. Six. Seven."

    # Spacy based - potentially, more accurate
    splitter_spacy = TextSplitter(n_words = 10, tokenizer=spacy.load('en_core_web_lg'))
    splitter_spacy.fit_transform(text)

    # NLTK-based - much, much faster
    splitter_nltk = TextSplitter(n_words = 10, tokenizer=tokenize)
    splitter_nltk.fit_transform(text)

    # Both result in 'This is an egg. Is the bird a chicken too?'
    """

    def __init__(self, n_words, tokenizer) -> None:

        if not isinstance(n_words, int) or n_words <= 0:
            raise ValueError("n_words needs to be a positive integer.")
        self.n_words = n_words
        # That one gets populated after a fit
        self.text = None
        # check if the tokenizer is supported.
        if not isinstance(tokenizer, types.ModuleType) and not isinstance(
            tokenizer, spacy.lang.en.English
        ):
            raise ValueError("tokenizer must be either a spacy model or nltk.tokenize")
        self.nlp = tokenizer

    def fit(self, X, y=None):
        """X is a string. This method is there mainly for sklearn compatibility and does not do much."""
        self.text = X
        return self

    def transform(self, X):
        """X is a string. Returns transformed and cut X."""
        if X != self.text:
            raise ValueError(
                "This transformer should not be used with different string inputs."
            )

        if len(X.split(" ")) <= self.n_words:
            return X

        sentences = self.split_in_sentences()
        if len(sentences[0].split(" ")) >= self.n_words:
            return sentences[0]
        start_string = self.add_sentences_from_the_start(
            sentences, int(self.n_words / 2)
        )
        end_string = self.add_sentences_from_the_end(sentences, int(self.n_words / 2))
        X_transformed = start_string + " " + end_string
        return X_transformed

    def split_in_sentences(self):
        """Helper function, but can be used as a standalone if splitting in sentences is needed."""
        if self.text is None:
            raise NotFittedError(
                "TextSplitter has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        if isinstance(self.nlp, spacy.lang.en.English):
            doc = self.nlp(self.text)
            list_of_sentences = [sent.text.strip() for sent in doc.sents]
        else:
            list_of_sentences = self.nlp.sent_tokenize(self.text)
        return list_of_sentences

    def _count_tokens(self, some_string):
        """Helper function to count the number of tokens in a string."""

        prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
        some_string = re.sub(prefixes, "\\1", some_string)

        if isinstance(self.nlp, spacy.lang.en.English):
            n_tokens = len(self.nlp(some_string.replace(".", "").lstrip(" ").rstrip(" ")))
        else:
            n_tokens = len(self.nlp.word_tokenize(some_string.replace(".", "")))
        return n_tokens

    def add_sentences_from_the_start(self, list_of_sentences, target_words):
        """Helper function that keeps adding sentences picked from the start of a string."""
        transformed_text = ""
        i = 0
        while self._count_tokens(transformed_text) < target_words:
            try:
                transformed_text = transformed_text + " " + list_of_sentences[i]
                i += 1
            except IndexError:
                break
        return transformed_text.lstrip(" ")

    def add_sentences_from_the_end(self, list_of_sentences, target_words):
        """Helper function that keeps adding sentences picked from the back of a string."""
        transformed_text = ""
        i = -1
        while self._count_tokens(transformed_text) < target_words:
            try:
                transformed_text = list_of_sentences[i] + " " + transformed_text
                i += -1
            except IndexError:
                break
        return transformed_text.rstrip(" ")
