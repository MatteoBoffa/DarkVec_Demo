import gzip
import csv
from collections import Counter


def count_daily_frequency(trace_file, min_pkts):
    """
    Count source IP occurrence frequency in a gzipped CSV trace file and
    return only those IPs whose daily packet count exceeds a given threshold.

    This function scans a compressed CSV file line by line without loading
    it into memory, making it efficient for large trace datasets. It assumes
    that the source IP address is stored in the 4th column (index 3).

    Parameters
    ----------
    trace_file : str
        Path to the gzipped CSV file containing the trace data. The file
        must be readable with :func:`gzip.open` and follow a row structure
        where the 4th column contains the source IP.
    min_pkts : int
        Minimum number of packets that a source IP must have sent during
        the day to be retained. IPs with counts **strictly greater than**
        this threshold are returned.

    Returns
    -------
    list of str
        List of source IPs whose daily packet count exceeds ``min_pkts``.
        The returned list may be used as a filter for downstream processing,
        such as corpus construction or sequence extraction.

    Notes
    -----
    - Uses :class:`collections.Counter` for efficient frequency counting.
    - Processes the file in streaming mode to avoid high memory usage.
    - Prints summary statistics about dropped and retained IPs.

    Examples
    --------
    >>> filtered_ips = count_daily_frequency("trace_20240101.csv.gz", 50)
    >>> len(filtered_ips)
    1234
    """
    print("[FILTER] Extracting the filter...")

    counts = Counter()

    # Read gzipped CSV directly
    with gzip.open(trace_file, "rt", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 3:
                src_ip = row[3]
                counts[src_ip] += 1

    daily_frq = [ip for ip, c in counts.items() if c > min_pkts]

    print(f"         Dropped IPs sending less than {min_pkts} daily packets")
    print(f"         Retained IPs: {len(daily_frq)}")

    return daily_frq
