from articlevectorizer.preprocess import ColumnGrabber
from articlevectorizer.preprocess import StringCutter
from sklearn.pipeline import make_pipeline
import pandas as pd

import pytest

example = pd.DataFrame(
    {
        "text_column": [
            "This is a short and easy example string.",
            "This is another short and easy example string",
            " ",
        ],
        "non_text_column": [1, 2, 3],
    }
)
example_list = [
    "This is a short and easy example string.",
    "This is another short and easy example string",
    " ",
]


@pytest.mark.parametrize(
    "method,score",
    [
        ("split", ["This ... string.", "This ... string", " "]),
        ("first_n", ["This is", "This is", " "]),
        ("last_n", ["example string.", "example string", " "]),
    ],
)
def test_simple_pipeline_usecase(method, score):
    cutter = make_pipeline(
        ColumnGrabber("text_column"), StringCutter(method=method, n_words=2)
    )
    assert cutter.transform(example) == score


@pytest.mark.parametrize(
    "method,score",
    [
        ("split", ["This ... string.", "This ... string", " "]),
        ("first_n", ["This is", "This is", " "]),
        ("last_n", ["example string.", "example string", " "]),
    ],
)
def test_simple_standalone_usecase(method, score):
    cutter = StringCutter(method=method, n_words=2)
    assert cutter.transform(example_list) == score


def test_raise_error():
    with pytest.raises(ValueError):
        cutter = make_pipeline(
            ColumnGrabber("text_column"),
            StringCutter(method="not_available", n_words=2),
        )
