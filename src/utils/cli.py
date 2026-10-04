"""Command-line helpers."""


def parse_csv(arg, default=None):
    """Split a comma-separated argument into stripped items; `default` when it is None."""
    if arg is None:
        return default
    return [x.strip() for x in arg.split(',')]
