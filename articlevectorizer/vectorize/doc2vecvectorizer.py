import logging
import gensim
import numpy as np
from articlevectorizer.base import VectorizerBase
from sklearn.exceptions import NotFittedError


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class Doc2VecVectorizer(VectorizerBase):
    """
    A Doc2Vec vectorizer compatible with sklearn.
    Only main params are predefined, but all other compatible doc2vec features can be supplied via **kwargs.

    Params:
    vector_size : int, optional
        Dimensionality of the feature vectors.
    dm : int {1,0}, optional
        Defines the training algorithm. If `dm=1` - distributed memory (PV-DM) is used.
        Otherwise, distributed bag of words (PV-DBOW) is employed.
    min_count : int, optional
        Ignores all words with total frequency lower than this.
    epochs : int, optional
        Number of epochs to iterate through the corpus.

    Usage example:

    d2v = Doc2VecVectorizer()
    d2v.fit(train.title)
    vec = d2v.transform(train.head(3).title)

    In case other doc2vec params are required, like window or alpha, they can still be supplied

    d2v = Doc2VecVectorizer(window = 10)
    d2v.fit(train.title)
    vec = d2v.transform(train.head(3).title)

    """

    def __init__(self, vector_size=300, min_count=2, epochs=100, dm=1, **kwargs):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.gensim_model = None
        self.other_gensim_args = kwargs

    def fit(self, X, y=None):
        """Fit a doc2vec model, incl preprocessing. X can be a list or a pandas.Series or other iterable."""

        X_transformed = [
            gensim.models.doc2vec.TaggedDocument(
                gensim.utils.simple_preprocess(article), [i]
            )
            for i, article in enumerate(X)
        ]

        doc2vec_model = gensim.models.doc2vec.Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            **self.other_gensim_args
        )

        doc2vec_model.build_vocab(X_transformed)
        doc2vec_model.train(
            X_transformed,
            total_examples=doc2vec_model.corpus_count,
            epochs=doc2vec_model.epochs,
        )
        self.gensim_model = doc2vec_model
        return self

    def transform(self, X):
        """Input is a list of texts. Returns a numpy array of vectors."""
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        transformed_data = [gensim.utils.simple_preprocess(doc) for doc in X]
        vectors = [self.gensim_model.infer_vector(doc) for doc in transformed_data]
        return np.array(vectors)
