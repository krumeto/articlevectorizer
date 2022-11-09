from articlevectorizer.base import VectorizerBase


class StringCutter(VectorizerBase):
    """
    Component that can grabs a list of strings and returns a portion of the string - either the first n words, last n words
    or a concat of the start and end of a string.

    It is helpful when experimenting with transformers which take only shorter strings as inputs.
    Many of the interfaces just handle long texts by cutting the first n (usually - 512) tokens.
    This class allows to get the end of the sequence or concat part of the start and part of the end of the string.

    ##############################
    Example:
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from data_transformers import ColumnGrabber, StringCutter

    example = pd.DataFrame({
        "text_column": ["This is an example string.", "This is another one", " "],
        "non_text_column": [1,2,3]
        })

    cutter_pipe = make_pipeline(ColumnGrabber(colname = "text_column"), StringCutter(method = 'first_n', n_words = 10))
    cutter_pipe.transform(example)
    """

    def __init__(self, method, n_words) -> None:

        valid_methods = ["first_n", "last_n", "split"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.method = method

        if not isinstance(n_words, int) or n_words <= 0:
            raise ValueError(
                f"n_words needs to be a positive integer for method {self.method}"
            )
        self.n_words = n_words

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        Depending on the method chosen, it can return the full string ('full'),
        the first n words ('first_n'), the last n words ('last_n') or split the n_words between the start and end of string ('split')
        """
        self._validate_list(X)

        if self.method == "first_n":
            interim = [x.split(" ") for x in X]
            results = [" ".join(x[: self.n_words]) for x in interim]
        if self.method == "last_n":
            interim = [x.split(" ") for x in X]
            results = [" ".join(x[-self.n_words :]) for x in interim]
        if self.method == "split":
            interim = [x.split(" ") for x in X]
            # Get half from start and half from end and concat with an [EOS] token in between.
            # if the string is shorter than n_words, return the string
            results = [
                " ".join(x[: int(self.n_words / 2)])
                + " ... "
                + " ".join(x[-int(self.n_words / 2) :])
                if self.n_words < len(x)
                else " ".join(x)
                for x in interim
            ]

        return results

    @staticmethod
    def _validate_list(obj):
        # verify this is a list
        if not isinstance(obj, list):
            raise AttributeError("Input to transform must be a list")
