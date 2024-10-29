def binary_reverse(targets, labels):
    return [labels[0] if target == labels[1] else labels[1] for target in targets]


def string_to_float(string, default=-1.0):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def check_data_state(preds, targets):
    assert len(preds) == len(targets)
