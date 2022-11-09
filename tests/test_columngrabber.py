from articlevectorizer.preprocess.columngrabber import ColumnGrabber
import pandas as pd

example = pd.DataFrame(
    {
        "text_column": ["This is an example string.", "This is another one", " "],
        "non_text_column": [1, 2, 3],
    }
)


def test_results():
    assert ColumnGrabber(colname="text_column").transform(example) == [
        "This is an example string.",
        "This is another one",
        " ",
    ]


def test_return_type():
    assert isinstance(ColumnGrabber(colname="text_column").transform(example), list)
