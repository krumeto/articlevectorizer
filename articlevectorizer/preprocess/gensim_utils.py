import gensim
from articlevectorizer.preprocess.columngrabber import ColumnGrabber


def df_to_gensim_corpus(dataf, column):
    """
    Input is a dataframe and a column. Output is a python generator in the prefered gensim format.

    Usage:
    transformed_corpus = board.pin_read("articles_train").pipe(df_to_gensim_corpus, 'article_fulltext')
    # Frequently needed as list, then
    transformed_corpus = list(transformed_corpus)

    doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    doc2vec_model.build_vocab(transformed_corpus)
    doc2vec_model.train(transformed_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)


    """
    data_list = ColumnGrabber(colname=column).transform(dataf)
    for i, article in enumerate(data_list):
        tokens = gensim.utils.simple_preprocess(article)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
