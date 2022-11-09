from articlevectorizer.vectorize import SpacyVectorizer
import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

input = [
    "I like cats and some dogs.",
    "World politics is my cup of tea but I also watch cricket."
]

model_names = ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"]

# Simple shape test
def test_output_shape(input=input):
    n_sentences = len(input)
    output = SpacyVectorizer().transform(input)
    assert n_sentences == output.shape[0]


# Sentence similarity test, indexes are equal to the index of the input list of strings.


@pytest.mark.parametrize(
    "new_sentence,expectation_index",
    [
        (["This mouse is so cute. Mouses are the best. I do not like cats at all."], 0),
        (["The president is very diplomatic."], 1),
    ],
)
def test_most_similar_doc(new_sentence, expectation_index, original_data=input):
    """Ensure most similar sentences are the obviously correct ones."""
    model = SpacyVectorizer()
    new_doc_vector = model.transform(new_sentence)
    vectorized_corpus = model.transform(original_data)
    sim = cosine_similarity(X=vectorized_corpus, Y=new_doc_vector)
    argmax = np.argmax(sim)
    assert original_data[argmax] == original_data[expectation_index]
