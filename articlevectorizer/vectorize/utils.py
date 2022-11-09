import numpy as np
import pandas as pd
import pins

def load_vectors(board, vector_name, return_type = 'numpy'):
    """A function to load vectors."""
    valid_return_types = ['numpy','pandas', 'dict']
    if return_type not in valid_return_types:
        raise ValueError(f"Return type must be one of {valid_return_types}")

    embedding_dict = board.pin_read(vector_name)
    if return_type == 'dict':
        return embedding_dict
    embedding_df = pd.DataFrame(embedding_dict).T
    if return_type == 'pandas':
        return embedding_df
    corpus_embeddings = embedding_df.to_numpy()
    if return_type == 'numpy':
        return corpus_embeddings


def list_of_cluster_items_to_dict(list_of_cluster_items, cluster_name):
    """
    Input is a list of indexes - numbers or article ids.
    Output is a dict where every id is the key and every value is the cluster_name
    """
    list_of_cluster_names = [str(cluster_name)]*len(list_of_cluster_items)
    return dict(zip(list_of_cluster_items, list_of_cluster_names))


def dict_of_cluster_items_to_df(dict_of_cluster_items, column_name = "clusters"):
    """
    Small util to turn dict to pandas DataFrame.
    """
    clusters_df = pd.DataFrame(dict_of_cluster_items, index = [0]).T
    clusters_df.columns = [column_name]
    clusters_df = clusters_df.sort_index()

    return clusters_df

def rollup_low_frequency_desks(dataf, column = 'editorial_desk', threshold = 200):
    """A function to rollup smaller desks into an Others category"""
    df_copy = dataf.copy()
    desk_counts = df_copy[column].value_counts(dropna=False)
    vals_to_remove = desk_counts[desk_counts <= threshold].index.values
    df_copy[column].loc[df_copy[column].isin(vals_to_remove)] = "Others"
    return df_copy