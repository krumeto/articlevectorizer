import spacy
from articlevectorizer.base import VectorizerBase


class LemmatizerSpacy(VectorizerBase):
    """A class to lemmatize a list of strings using spaCy's lemmatizer.
    
    It can be particularly useful for Tf-Idf as lemmatizing would significantly decrease
    the size of the Tf-Idf vector (while still keeping performance relatively good.)
    Arguments:
    - spacy_model: name of the spacy model, default is 'en_core_web_lg'. For list of all spaCy models, see 
    https://spacy.io/models/en
    - n_process: None or int, if running in parallel, number of parallel python processes. 
    
    Example:
        examples = ["I am the worst painter in the world.", "Cats don't like swimming. But they can swim. They just prefer not to."]

        lem = LemmatizerSpacy()
        print(lem.transform(examples))
        # ['I be the bad painter in the world.', 'cat donot like swim. but they can swim. they just prefer not to.']
    """
    def __init__(self, spacy_model = 'en_core_web_lg', n_process = None):
        self.spacy_model = spacy.load(spacy_model)

        if (not isinstance(n_process, int)) and (n_process is not None):
            raise AttributeError("n_process must be either int or None")

        self.n_process = n_process
        self.spacy_model.get_pipe("lemmatizer")
        
    def transform(self, X, y=None):
        """
        Transforms a list of strings into a list of lemmatized strings.
        """
        # this is a python generator
        if self.n_process is not None:
            spacy_docs = self.spacy_model.pipe(X, n_process = self.n_process)
        else:
            spacy_docs = self.spacy_model.pipe(X)

        lemmatized_text = list()
        for article in spacy_docs:
            # Spacy can lemmatize full sentences
            # The result is a list of lemmatized sentences per article
            list_view = [sentence.lemma_ for sentence in article.sents]
            # Join the sentences back to a full string
            lemmatized_text.append(" ".join(list_view))

        return lemmatized_text





