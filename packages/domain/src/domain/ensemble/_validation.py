"""Private input validation for ensemble calculation modules.

Ensemble statistics and probability functions operate on a flat, finite
sequence of member values. ``_coerce_members`` normalizes any supported input
to a ``numpy.float64`` array so downstream math is deterministic and never
silently consumes malformed data.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from domain.exceptions import EmptyEnsembleError, InvalidEnsembleError


def _is_numeric_scalar(value: object) -> bool:
    """Return whether *value* is a numeric scalar (not a bool or string).

    Booleans are excluded even though Python treats ``bool`` as a subclass of
    ``int``; ``True``/``False`` are not ensemble member values.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _has_only_numeric_elements(
    members: Sequence[object] | npt.NDArray[np.float64],
) -> bool:
    """Return whether every element of ``members`` is a numeric scalar.

    String, byte, boolean, and arbitrary-object elements are rejected so the
    element-type check runs *before* the ``float64`` conversion, preventing
    silent coercion of values such as ``"1.5"`` or ``True``.
    """
    if isinstance(members, np.ndarray):
        if members.dtype == np.bool_:
            return False
        if np.issubdtype(members.dtype, np.number):
            return True
        # Non-numeric dtype (e.g. object/string): validate each element.
        return all(_is_numeric_scalar(value) for value in members.flat)
    return all(_is_numeric_scalar(value) for value in members)


def _coerce_members(
    members: Sequence[float | int] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Validate and normalize an ensemble member sequence.

    Args:
        members: Ensemble member values as a sequence of ints/floats or a
            NumPy array.

    Returns:
        A one-dimensional ``numpy.float64`` array of the member values.

    Raises:
        EmptyEnsembleError: If the sequence is empty.
        InvalidEnsembleError: If the input is not a one-dimensional numeric
            sequence, contains non-numeric (e.g. boolean or string) elements,
            or contains non-finite values.
    """
    if not isinstance(members, np.ndarray) and (
        not isinstance(members, Sequence) or isinstance(members, (str, bytes, bytearray))
    ):
        raise InvalidEnsembleError(
            "ensemble members must be a sequence of numeric values, "
            f"got {type(members).__name__}"
        )

    if not _has_only_numeric_elements(members):
        raise InvalidEnsembleError(
            "ensemble members must be a sequence of numeric values, "
            "got elements of a non-numeric type (e.g. bool or string)"
        )

    array = np.asarray(members, dtype=np.float64)

    if array.ndim != 1:
        raise InvalidEnsembleError(
            f"ensemble members must be one-dimensional, got {array.ndim} dimensions"
        )
    if array.size == 0:
        raise EmptyEnsembleError("ensemble members must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise InvalidEnsembleError("ensemble members must contain only finite numeric values")

    return array
