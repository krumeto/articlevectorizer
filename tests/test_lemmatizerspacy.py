from articlevectorizer.preprocess.lemmatizer import LemmatizerSpacy

examples = ['I am the worst painter in the world.', "Cats don't like swimming. But they can swim. They just prefer not to.", ""]

def test_outcome():
    lem = LemmatizerSpacy()
    assert lem.transform(examples) == ['I be the bad painter in the world.', 'cat donot like swim. but they can swim. they just prefer not to.', '']


def test_outcome_type():
    lem = LemmatizerSpacy()
    assert isinstance(lem.transform(examples), list)