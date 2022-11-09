from articlevectorizer.vectorize.sentencetransformer import (
    SentenceTransformerVectorizer,
)
import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

input = [
    "I like cats and some dogs.",
    "World politics is my cup of tea but I also watch cricket.",
    " ",
    "Mars and Venus are not that far from the Earth.",
]

similar_input = [
    "This mouse is so cute.",
    "The president is very diplomatic.",
    "   ",
    "The rings of Saturn are spectacular.",
]

model_names = ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"]

# Simple shape test
def test_output_shape(input=input):
    n_sentences = len(input)
    output = SentenceTransformerVectorizer("all-MiniLM-L6-v2").transform(input)
    assert n_sentences == output.shape[0]


# Sentence similarity test, zipping two lists for which each pair of sentences are obviously the closest match.
# We would like transformers to produce similar vectors for similar texts.


@pytest.mark.parametrize("new_sentence,expectation", list(zip(similar_input, input)))
def test_most_similar_doc(new_sentence, expectation, original_data=input):
    """Ensure most similar sentences are the obviously correct ones."""
    model = SentenceTransformerVectorizer("all-MiniLM-L6-v2")
    new_doc_vector = model.transform([new_sentence])
    vectorized_corpus = model.transform(original_data)
    sim = cosine_similarity(X=vectorized_corpus, Y=new_doc_vector)
    argmax = np.argmax(sim)
    assert original_data[argmax] == expectation
