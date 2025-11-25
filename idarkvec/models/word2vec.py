from gensim.models import Word2Vec as W2V
from multiprocessing import cpu_count
import pandas as pd
import numpy as np


class Word2Vec:
    """Wrapper around :class:`gensim.models.Word2Vec` for training,
    updating, and exporting embeddings.

    This class provides a simplified interface for:

    - Training a new Word2Vec model from a tokenized corpus.
    - Updating an existing model from disk.
    - Exporting embeddings as a pandas DataFrame (optionally labeled).
    - Deleting specific embeddings from the model.

    Parameters
    ----------
    c : int, optional
        Context window size for the Word2Vec model (``window`` argument).
        Default is 25.
    e : int, optional
        Embedding dimensionality for the Word2Vec model
        (``vector_size`` argument). Default is 50.
    epochs : int, optional
        Number of training epochs. Default is 1.
    source : str, optional
        Base path (without extension) of an existing Word2Vec model to
        load. The file is expected to be named ``"{source}.word2vec"``.
        If ``None``, no model is loaded at initialization. Default is
        ``None``.
    destination : str, optional
        Base path (without extension) used when saving models or
        embeddings. The file will be named ``"{destination}.word2vec"``
        or ``"{destination}.csv.gz"`` for embeddings. Default is
        ``None``.
    seed : int, optional
        Random seed used for Word2Vec training, to ensure
        reproducibility. Default is 15.
    """

    def __init__(self, c=25, e=50, epochs=1, source=None, destination=None, seed=15):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.seed = seed
        self.source = source
        self.destination = destination

        if type(source) != type(None):
            self.model = W2V.load(f"{self.source}.word2vec")
        else:
            self.model = None

    def train(self, corpus, save=False):
        """Train a new Word2Vec model from a tokenized corpus.

        This method always creates a *new* model, ignoring any model
        previously loaded at initialization.

        Parameters
        ----------
        corpus : iterable of list of str
            Tokenized corpus used to train the model. Each element is a
            sentence (or document) represented as a list of tokens.
        save : bool, optional
            If ``True``, the trained model is saved to
            ``"{destination}.word2vec"``. Default is ``False``.
        """
        print(f"[WORD2VEC] Training a new word2vec model...")
        self.model = W2V(
            sentences=corpus,
            vector_size=self.embedding_size,
            window=self.context_window,
            epochs=self.epochs,
            workers=cpu_count(),
            min_count=0,
            sg=1,
            negative=5,
            sample=0,
            seed=self.seed,
        )
        print(f"           {self.destination}.word2vec trained")
        print(f"           Total embeddings: {len(self.model.wv.index_to_key)}")
        if save:
            self.model.save(f"{self.destination}.word2vec")
            print(f"           Model {self.destination}.word2vec saved")

    def update(self, corpus, save=False):
        """Update an existing Word2Vec model with additional corpus data.

        This method assumes that ``self.model`` has already been loaded
        or trained. It extends the model’s vocabulary and retrains it
        on the provided corpus.

        Parameters
        ----------
        corpus : iterable of list of str
            Additional tokenized corpus used to update the model.
        save : bool, optional
            If ``True``, the updated model is saved to
            ``"{destination}.word2vec"``. Default is ``False``.
        """
        print(f"[WORD2VEC] Updating {self.source}.word2vec model ...")
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(
            corpus, total_examples=self.model.corpus_count, epochs=self.epochs
        )
        print(f"           {self.destination}.word2vec trained")
        print(f"           Total embeddings: {len(self.model.wv.index_to_key)}")
        if save:
            self.model.save(f"{self.destination}.word2vec")
            print(f"           Model {self.destination}.word2vec saved")

    def get_embeddings(self, ips=None, labels=None, dst_path=None):
        """Export embeddings as a pandas DataFrame.

        Parameters
        ----------
        ips : list of str, optional
            Tokens (e.g., IP addresses) for which embeddings should be
            retrieved. If ``None``, embeddings for all tokens in the
            model vocabulary are used, and the index is
            ``model.wv.index_to_key``. Default is ``None``.
        labels : pandas.DataFrame or list, optional
            Optional labels to attach to each embedding.

            - If a pandas DataFrame, it must contain an ``"ip"`` column
              (matching the index) and a ``"class"`` column. A ``class``
              column is added to the result; missing labels are filled
              with ``"unknown"``.
            - If a list, it must have the same length as ``ips``. This
              branch currently only validates the length and raises if
              lengths mismatch or if ``ips`` is ``None``.
        dst_path : str, optional
            Base path (without extension) where the resulting DataFrame
            is saved as ``"{dst_path}.csv.gz"`` if provided. Default is
            ``None``.

        Returns
        -------
        pandas.DataFrame
            DataFrame of embeddings. Rows correspond to tokens (indexed
            by ``ips``) and columns to embedding dimensions. If labels
            are provided as a DataFrame, an additional ``"class"``
            column is appended.
        """
        # Return a dataframe with the provided IPs as index
        # If provided also labels they are set as last column named `class`
        # Labels must be provided as another pandas dataframe with a column
        # named `ip` and a column named `class`
        if type(ips) == type(None):
            ips = [x for x in self.model.wv.index_to_key]
        embeddings = self.model.wv.vectors
        embeddings = pd.DataFrame(embeddings, index=ips)

        if type(labels) == pd.core.frame.DataFrame:
            embeddings = (
                embeddings.reset_index()
                .rename(columns={"index": "ip"})
                .merge(labels, on="ip", how="left")
                .set_index("ip")
                .fillna("unknown")
            )
        elif type(labels) == list:
            if type(ips) == type(None):
                raise ValueError(f"Providing labels requires also ips")
            elif len(labels) != len(ips):
                raise ValueError(f"Length mismatch:")

        if type(dst_path) != type(None):
            embeddings.to_csv(f"{dst_path}.csv.gz")

        return embeddings

    def del_embeddings(self, to_drop, dst_path=None):
        """Delete specific embeddings from the underlying Word2Vec model.

        This method removes the given tokens from:

        - The vocabulary (``wv.vocab``).
        - The list of tokens (``wv.index2word``).
        - The embedding matrix (``wv.vectors``).
        - The negative sampling weights (``trainables.syn1neg``).

        It also reindexes the remaining tokens to keep indices
        consistent.

        Parameters
        ----------
        to_drop : list of str
            Tokens (e.g., IP addresses) whose embeddings should be
            removed from the model.
        dst_path : str, optional
            Base path (without extension) where the modified model is
            saved as ``"{dst_path}.word2vec"`` if provided. Default is
            ``None``.
        """
        idx = np.isin(self.model.wv.index2word, to_drop)
        idx = np.where(idx == True)[0]
        self.model.wv.index2word = list(
            np.delete(self.model.wv.index2word, idx, axis=0)
        )
        self.model.wv.vectors = np.delete(self.model.wv.vectors, idx, axis=0)
        self.model.trainables.syn1neg = np.delete(
            self.model.trainables.syn1neg, idx, axis=0
        )
        list(
            map(
                self.model.wv.vocab.__delitem__,
                filter(self.model.wv.vocab.__contains__, to_drop),
            )
        )

        for i, word in enumerate(self.model.wv.index2word):
            self.model.wv.vocab[word].index = i

        if type(dst_path) != type(None):
            self.model.save(f"{dst_path}.word2vec")
