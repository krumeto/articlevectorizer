import spacy
import numpy as np
from articlevectorizer.base import VectorizerBase


class SpacyVectorizer(VectorizerBase):
    """A class to use spaCy to create vectors per document.

    Spacy returns the average of word vectors as a document vector.

    Arguments:
    - spacy_model: name of the spacy model, default is 'en_core_web_lg'. For list of all spaCy models, see 
    https://spacy.io/models/en
    - n_process: None or int, if running in parallel, number of parallel python processes. 
    
    Example:
        examples = ["I am the worst painter in the world.", "Cats don't like swimming. But they can swim. They just prefer not to."]

        vectorizer = SpacyVectorizer()
        vectors = vectorizer.transform(examples)

    """
    def __init__(self, spacy_model = 'en_core_web_lg', n_process = None):
        self.spacy_model = spacy.load(spacy_model)

        if (not isinstance(n_process, int)) and (n_process is not None):
            raise AttributeError("n_process must be either int or None")

        self.n_process = n_process
        
    def transform(self, X, y=None):
        """
        Transforms a list of strings into a numerical vectors.
        """
        # this is a python generator
        if self.n_process is not None:
            spacy_docs = self.spacy_model.pipe(X, disable=['ner', 'parser'], n_process = self.n_process)
        else:
            spacy_docs = self.spacy_model.pipe(X, disable=['ner', 'parser'])

        spacy_vectors = [x.vector for x in spacy_docs]

        return np.array(spacy_vectors)

    def get_vector_norm(self, X):
        """Get the L2 norm of the spacy vectors."""
        if self.n_process is not None:
            spacy_docs = self.spacy_model.pipe(X, n_process = self.n_process)
        else:
            spacy_docs = self.spacy_model.pipe(X)

        spacy_vectors_norms = [x.vector_norm for x in spacy_docs]

        return np.array(spacy_vectors_norms)
