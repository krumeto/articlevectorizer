from articlevectorizer.base import VectorizerBase
from sentence_transformers import SentenceTransformer


class SentenceTransformerVectorizer(VectorizerBase):
    """
    Create a SentenceTransformer.
    You can find the available options here:
    https://www.sbert.net/docs/pretrained_models.html#model-overview
    Arguments:
    - name: name of model, see available options
    - max_seq_length: 'default' or int, typically up to 509 (leaving tokens for beginning and end)

    Example usage:
    model_to_use = 'all-mpnet-base-v2'
    sent_transf = SentenceTransformerVectorizer(model_to_use)
    # Get the max_seq_length maintained
    print(sent_transf.tfm.max_seq_length)

    input = ["short sentence", 'long long long sentence']
    output = sent_transf.transform(input)
    # Returns a numpy matrix of size (2, 768)
    print(output.shape)
    """

    def __init__(self, name, max_seq_length="default"):
        self.name = name
        self.tfm = SentenceTransformer(name)

        if (
            (not isinstance(max_seq_length, int))
            and (max_seq_length != "default")
            and (max_seq_length > 512)
        ):
            raise AttributeError(
                "max_seq_length must be either 'default' or an integer 512 or smaller."
            )

        # setting max sequence length to the input if not default
        if isinstance(max_seq_length, int):
            self.max_seq_length = max_seq_length
            self.tfm.max_seq_length = max_seq_length

    def transform(self, X, y=None):
        """Transforms the text (X) list of string into a numeric representation. Out is a numpy array"""
        return self.tfm.encode(X)


def print_main_sentence_transformers():
    """Just a storage of main models to try for experiment.
    To get more/all, go to https://www.sbert.net/docs/pretrained_models.html#model-overview
    """
    main_models = [
        "all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "multi-qa-distilbert-cos-v1",
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-albert-small-v2",
        # Below are not specifically for sentence transformations
        "gtr-t5-large",
        "gtr-t5-base",
        "sentence-t5-large",
        # Just averaging word embeddings
        "average_word_embeddings_komninos",
        "average_word_embeddings_glove.6B.300d",
    ]
    return main_models
