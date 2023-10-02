class Dimensionality:
    """
    Dummy class to allow skipping clustering during model fitting.
    """

    def __init__(self, reduced_embeddings):
        """
        Initialize the Dimensionality class.

        Args:
            reduced_embeddings (array-like): Embeddings after dimensionality reduction.
        """
        self.reduced_embeddings = reduced_embeddings

    def fit(self, X):
        """
        Dummy fit method.

        Args:
            X (array-like): Input data.

        Returns:
            self: An instance of the class.
        """
        return self

    def transform(self, X):
        """
        Dummy transform method.

        Args:
            X (array-like): Input data.

        Returns:
            array-like: Reduced embeddings.
        """
        return self.reduced_embeddings