from sklearn.base import BaseEstimator, TransformerMixin


class VectorizerBase(BaseEstimator, TransformerMixin):
    """Base class for feature transformers"""

    def fit(self, X, y=None):
        """No-op."""
        return self

    def partial_fit(self, X, y=None):
        """No-op."""
        return self
