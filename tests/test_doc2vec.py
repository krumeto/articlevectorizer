from articlevectorizer.vectorize import Doc2VecVectorizer
import pytest
import gensim
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
import pickle

# Weird MAC issue with ssl
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

train, _ = fetch_20newsgroups(
    random_state=1,
    subset="train",
    remove=("headers", "footers", "quotes"),
    return_X_y=True,
)

input = [
    "The price of the Microsoft stocks dropped.",
    "Rome is the capital of Italy.",
    "Coal leads to air pollution. ",
]

similar_input = [
    "The price of the Unilever stocks dropped.",
    "Paris is the capital of France.",
    "Gas leads to air pollution. ",
]

model = Doc2VecVectorizer(vector_size=30, epochs=5)
model.fit(train)

# Simple shape test
def test_output_shape(input=input, model=model):
    n_sentences = len(input)
    output = model.fit_transform(input)
    assert n_sentences == output.shape[0]


# Below tests inspired by doc2vec 3.8.3 version tests for doc2vec.

# Test setting params
def testSetGetParams(model=model):
    # updating only one param
    model.set_params(epochs=6)
    model_params = model.get_params()
    assert model_params["epochs"] == 6

    # verify that the attributes values are also changed for `gensim_model` after fitting
    model.fit(input)
    assert getattr(model.gensim_model, "epochs") == 6


# Test if the model works in a sklearn pipeline
def testPipeline(model=model):
    np.random.seed(0)  # set fixed seed to get similar values everytime

    class_dict = {"mathematics": 1, "physics": 0}
    train_data = [
        (["calculus", "mathematical"], "mathematics"),
        (["geometry", "operations", "curves"], "mathematics"),
        (["natural", "nuclear"], "physics"),
        (["science", "electromagnetism", "natural"], "physics"),
    ]
    train_input = [" ".join(x[0]) for x in train_data]
    train_target = [class_dict[x[1]] for x in train_data]

    clf = LogisticRegression(penalty="l2", C=0.1)
    clf.fit(model.transform(train_input), train_target)
    text_w2v = Pipeline(
        [
            (
                "features",
                model,
            ),
            ("classifier", clf),
        ]
    )
    score = text_w2v.score(train_input, train_target)
    assert score > 0.40


# test if you can pickle and unpickle
def testPersistence(model=model, input=similar_input):
    model_dump = pickle.dumps(model)
    model_load = pickle.loads(model_dump)

    doc = [similar_input[0]]
    loaded_transformed_vecs = model_load.transform(doc)

    # sanity check for transformation operation
    assert loaded_transformed_vecs.shape[0] == 1
    assert loaded_transformed_vecs.shape[1] == model_load.vector_size

    # comparing the original and loaded models
    original_transformed_vecs = model.transform(doc)
    passed = np.allclose(
        sorted(loaded_transformed_vecs), sorted(original_transformed_vecs), atol=1e-1
    )
    assert passed == True


# Test if the same/similar to the original Gensim model without a wrapper
def testConsistencyWithGensimModel(input=input, similar_input=similar_input):
    # training a D2VTransformer
    model = Doc2VecVectorizer(min_count=1, vector_size=100, dm=1, epochs=5)
    model.fit(input)

    d2v_sentences = []
    for i, article in enumerate(input):
        tokens = gensim.utils.simple_preprocess(article)
        d2v_sentences.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))
    # training a Gensim Doc2Vec model with the same params
    gensim_d2vmodel = gensim.models.Doc2Vec(
        d2v_sentences, min_count=1, vector_size=100, dm=1, epochs=5
    )
    gensim_d2vmodel.build_vocab(d2v_sentences)
    gensim_d2vmodel.train(
        d2v_sentences,
        total_examples=gensim_d2vmodel.corpus_count,
        epochs=gensim_d2vmodel.epochs,
    )

    doc = similar_input[0]
    vec_transformer_api = model.transform(doc)  # vector returned by Doc2VecVectorizer
    vec_gensim_model = gensim_d2vmodel.infer_vector(
        gensim.utils.simple_preprocess(doc)
    )  # vector returned by Doc2Vec
    passed = np.allclose(vec_transformer_api, vec_gensim_model, atol=1e-1)
    assert passed == True


# Ensure you cannot call transform without fit before that.
def testModelNotFitted():
    model = Doc2VecVectorizer(min_count=1)
    with pytest.raises(Exception) as e_info:
        model.transform(1)
