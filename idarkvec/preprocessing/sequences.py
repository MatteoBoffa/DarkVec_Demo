class SequenceExtractor:
    """Utility class for extracting IP sequences grouped by destination
    port and protocol from a trace DataFrame.

    This class currently provides static helpers to transform a raw
    trace into ordered sequences of source IP addresses, grouped by
    the most frequently used (dst_port, proto) combinations.
    """

    @staticmethod
    def _extract_by_ports(df, top_ports):
        """Extract per-port sequences of source IPs from a trace.

        This method builds a combined *port/protocol* key for each row,
        selects the `top_ports` most frequent combinations, maps all
        remaining ones to a generic `"other"` bucket, and then builds
        ordered sequences of `src_ip` for each key.

        The expected columns in the input DataFrame are:

        - ``dst_port`` : destination port (numeric or string)
        - ``proto``    : transport protocol (e.g. TCP, UDP)
        - ``ts``       : timestamp, used to sort events chronologically
        - ``src_ip``   : source IP address

        Parameters
        ----------
        df : pandas.DataFrame
            Trace data containing at least the columns ``"dst_port"``,
            ``"proto"``, ``"ts"``, and ``"src_ip"``.
        top_ports : int
            Number of most frequent ``"port/proto"`` combinations to
            keep. All less frequent combinations are grouped under the
            label ``"other"``.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by the combined ``"port/proto"`` key
            (column ``"pp"``), with a single column:

            - ``"src_ip"``: a Python ``list`` of source IPs, ordered by
              timestamp.

            This structure is suitable for further processing into a
            corpus (e.g., dropping duplicates, feeding to Word2Vec).
        """
        df["pp"] = df["dst_port"].astype(str) + "/" + df["proto"].astype(str)
        topN = df.value_counts("pp").iloc[:top_ports].index
        df.loc[df[~df.pp.isin(topN)].index, "pp"] = "other"
        # Extract IPs sequences by ports
        sequences = (
            df.sort_values("ts").groupby("pp").agg({"src_ip": list}).sort_index()
        )

        return sequences
