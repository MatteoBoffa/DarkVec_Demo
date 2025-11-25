import numpy as np
import pandas as pd
from .filter import count_daily_frequency
from .sequences import SequenceExtractor


class BaseCorpus:
    """Base class for building a corpus of network service sequences
    from daily trace files.

    Each instance represents a single day's trace, which is loaded
    from a compressed CSV file and filtered by the minimum number
    of daily events per source IP.

    The typical workflow is:

    1. Filter the raw trace to keep only frequent source IPs.
    2. Extract per-IP sequences of services/ports (implemented in subclasses).
    3. Post-process these sequences to remove duplicates and build the corpus.

    Parameters
    ----------
    trace_path : str
        Base path of the trace files, without day suffix and extension.
        The actual file is expected to be named
        ``"{trace_path}_{day}.csv.gz"``.
    day : str or int
        Day identifier used to compose the filename. For example, if
        ``trace_path="trace"`` and ``day="20240101"``, the file
        ``"trace_20240101.csv.gz"`` will be loaded.
    min_freq : int
        Minimum number of daily occurrences per source IP. Only source
        IPs whose daily frequency is greater than or equal to this
        threshold are kept.

    Attributes
    ----------
    trace_path : str
        Base path of the trace files provided at initialization.
    day : str or int
        Day identifier used to select the trace file for this corpus.
    min_freq : int
        Minimum daily frequency used to filter source IPs.
    trace_file : str
        Full path to the compressed CSV trace file for the given day.
    """

    def __init__(self, trace_path, day, min_freq):
        self.min_freq = min_freq
        self.trace_path = trace_path
        self.day = day
        self.trace_file = f"{trace_path}_{day}.csv.gz"

    def _filter_trace(self):
        """Load the daily trace and filter it by source IP frequency.

        This method:

        1. Uses :func:`count_daily_frequency` to compute the number of
           events per source IP.
        2. Reads the raw trace from ``self.trace_file``.
        3. Keeps only rows whose ``src_ip`` appears at least
           ``self.min_freq`` times in the trace.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame containing only frequent source IPs,
            sorted by index.
        """
        _filter = count_daily_frequency(self.trace_file, self.min_freq)
        # Load raw trace
        df = pd.read_csv(self.trace_file, index_col=[0]).sort_index()
        # Apply filter
        df = df[df.src_ip.isin(_filter)]

        return df

    def _rearrange_sequences(self, sequences):
        """Rearrange per-IP service sequences into a corpus.

        Given a table (e.g., a DataFrame) of sequences, this method:

        1. Drops consecutive duplicate services within each sequence.
        2. Sorts sequences by their order identifier.
        3. Extracts the list of service sequences to form the final corpus.

        Parameters
        ----------
        sequences : pandas.DataFrame or pandas.Series
            Iterable of records (e.g., rows) where each item contains
            an order identifier and a sequence of services. It must be
            compatible with ``sequences.itertuples()`` and produce
            two fields: ``order`` and ``service``.

        Returns
        -------
        list of list of str
            Corpus represented as a list of service sequences, where
            each sequence is a list of tokens (services).
        """
        sequences = [self._drop_duplicates(x) for x in sequences.itertuples()]
        # Manage final corpus
        sequences.sort(key=lambda x: x[0])
        corpus = [x[1] for x in sequences]

        return corpus

    def _drop_duplicates(self, x):
        """Remove consecutive duplicate services from a sequence.

        Parameters
        ----------
        x : tuple
            Tuple with two elements: ``(order, service)``, where
            ``order`` is a sortable identifier (e.g., an integer)
            and ``service`` is an iterable of service tokens
            (e.g., list of str).

        Returns
        -------
        tuple
            A tuple ``(order, document)``, where ``document`` is a list
            of service tokens with consecutive duplicates removed.
        """
        order, service = x
        _prev = np.array(service)
        _next = np.roll(_prev, -1)
        _next[-1] = "NULL"
        document = _prev[_prev != _next]

        return (order, list(document))


class CorpusExtractor(BaseCorpus):
    """
    Methods
    -------
    from_darknet(top_ports, verbose=False)
        Build a corpus of darknet traffic sequences restricted to a
        given set of top ports.

    """

    __doc__ = BaseCorpus.__doc__ + __doc__

    def from_darknet(self, top_ports, verbose=False):
        """Extract a corpus of darknet service sequences.

        This method filters the daily trace, extracts per-IP sequences
        of services/ports associated with darknet traffic, and then
        rearranges them into a corpus suitable for downstream analysis
        (e.g., language modeling, clustering).

        Parameters
        ----------
        top_ports : iterable of int or str
            Collection of ports of interest (e.g., the most frequently
            observed ports). Only traffic involving these ports will be
            used to build sequences.
        verbose : bool, optional
            If ``True``, prints statistics about the extracted corpus,
            such as the number of raw and effective (unique) words.
            Default is ``False``.

        Returns
        -------
        list of list of str
            Corpus of darknet sequences, where each element is a list
            of tokens (services/ports) corresponding to an individual
            document/sequence.
        """
        trace = self._filter_trace()
        print(f"[CORPUS] Extracting the corpus...")
        ip_sequences = SequenceExtractor._extract_by_ports(trace, top_ports)
        corpus = self._rearrange_sequences(ip_sequences)

        if verbose:
            raw_words = np.hstack(corpus)
            effective_words = np.unique(raw_words)
            print(f"         Corpus extracted")
            print(f"         {raw_words.shape[0]} raw words")
            print(f"         {effective_words.shape[0]} effective words")

        return corpus
