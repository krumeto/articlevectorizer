from articlevectorizer.base import VectorizerBase
import pandas as pd


class ColumnGrabber(VectorizerBase):
    """
    Component that can grab a pandas column as a list.
    This can be useful when dealing with text encoders as these
    sometimes cannot deal with pandas columns.

    Base comes from Vincent WarmerDam (spaCy) - https://github.com/koaning/embetter/blob/main/embetter/grab.py

    ##############################
    Example:
    import pandas as pd
    from data_transformers import ColumnGrabber

    example = pd.DataFrame({
        "text_column": ["This is an example string.", "This is another one", " "],
        "non_text_column": [1,2,3]
        })

    grabber = ColumnGrabber(colname = "text_column")
    grabber.transform(example)
    """

    def __init__(self, colname) -> None:
        self.colname = colname

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        """
        self._validate_pandas(X)

        if self.colname not in X.columns.values:
            raise ValueError(
                f"No column named {self.colname}. Columns are {X.columns.values}"
            )

        return [x for x in X[self.colname]]

    @staticmethod
    def _validate_pandas(obj):
        # verify this is a DataFrame
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Input to transform must be a pandas DataFrame")
