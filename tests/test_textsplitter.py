from articlevectorizer.preprocess import TextSplitter
import pytest
import nltk
nltk.download('punkt')
from nltk import tokenize

import spacy

test_string = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't
hold with such nonsense.
Mr. Dursley was the director of a firm called Grunnings, which made drills.
"""
simple_string = (
    "This is an egg. This is a chicken. This is a bird. Is the bird a chicken too?"
)


# Test spacy
@pytest.mark.parametrize(
    "n_words,result",
    [
        (1, "This is an egg."),
        (9, "This is an egg. Is the bird a chicken too?"),
        (15, "This is an egg. This is a chicken. Is the bird a chicken too?"),
        (
            50,
            "This is an egg. This is a chicken. This is a bird. Is the bird a chicken too?",
        ),
    ],
)
def test_simple_usecase_spacy(n_words, result):
    splitter_spacy = TextSplitter(
        n_words=n_words, tokenizer=spacy.load("en_core_web_lg")
    )
    assert splitter_spacy.fit_transform(simple_string) == result


@pytest.mark.parametrize(
    "n_words,result",
    [
        (
            1,
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
        ),
        (
            25,
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. Mr. Dursley was the director of a firm called Grunnings, which made drills.",
        ),
        (
            50,
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't\nhold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills.",
        ),
        (150, test_string),
    ],
)
def test_complicated_usecase_spacy(n_words, result):
    splitter_spacy = TextSplitter(
        n_words=n_words, tokenizer=spacy.load("en_core_web_lg")
    )
    assert splitter_spacy.fit_transform(test_string) == result


# Test NLTK


@pytest.mark.parametrize(
    "n_words,result",
    [
        (1, "This is an egg."),
        (9, "This is an egg. Is the bird a chicken too?"),
        (15, "This is an egg. This is a chicken. Is the bird a chicken too?"),
        (50, simple_string),
    ],
)
def test_simple_usecase_nltk(n_words, result):
    splitter_nltk = TextSplitter(n_words=n_words, tokenizer=tokenize)
    assert splitter_nltk.fit_transform(simple_string) == result


@pytest.mark.parametrize(
    "n_words,result",
    [
        (
            1,
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
        ),
        (
            25,
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. Mr. Dursley was the director of a firm called Grunnings, which made drills.",
        ),
        (
            50,
            "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't\nhold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills.",
        ),
        (150, test_string),
    ],
)
def test_complicated_usecase_nltk(n_words, result):
    splitter_nltk = TextSplitter(n_words=n_words, tokenizer=tokenize)
    assert splitter_nltk.fit_transform(test_string) == result