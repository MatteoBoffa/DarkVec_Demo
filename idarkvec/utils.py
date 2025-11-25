from datetime import datetime, timedelta


def get_next_day(start):
    """Return the next day after a given YYYYMMDD date string.

    This utility converts a date in compact string format into a
    `datetime` object, adds one calendar day, and returns the result
    formatted back as a string in the same `YYYYMMDD` format.

    Parameters
    ----------
    start : str
        Date string in the format ``"YYYYMMDD"`` (e.g., ``"20240101"``).

    Returns
    -------
    str
        Date string representing the day after ``start`` in the format
        ``"YYYYMMDD"``.

    Examples
    --------
    >>> get_next_day("20240101")
    '20240102'
    >>> get_next_day("20231231")
    '20240101'
    """
    start = datetime.strptime(start, "%Y%m%d")
    day = start + timedelta(days=1)
    day = day.strftime("%Y%m%d")

    return day


def get_prev_day(start):
    """Return the previous day before a given YYYYMMDD date string.

    This utility converts a date in compact string format into a
    `datetime` object, subtracts one calendar day, and returns the result
    formatted back as a string in the same `YYYYMMDD` format.

    Parameters
    ----------
    start : str
        Date string in the format ``"YYYYMMDD"`` (e.g., ``"20240101"``).

    Returns
    -------
    str
        Date string representing the day before ``start`` in the format
        ``"YYYYMMDD"``.

    Examples
    --------
    >>> get_prev_day("20240101")
    '20231231'
    >>> get_prev_day("20240301")
    '20240229'
    """
    start = datetime.strptime(start, "%Y%m%d")
    day = start - timedelta(days=1)
    day = day.strftime("%Y%m%d")

    return day
