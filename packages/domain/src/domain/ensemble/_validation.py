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
            sequence, or contains non-finite values.
    """
    if not isinstance(members, np.ndarray) and (
        not isinstance(members, Sequence) or isinstance(members, (str, bytes, bytearray))
    ):
        raise InvalidEnsembleError(
            "ensemble members must be a sequence of numeric values, "
            f"got {type(members).__name__}"
        )

    try:
        array = np.asarray(members, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidEnsembleError(
            "ensemble members must be a sequence of numeric values"
        ) from exc

    if array.ndim != 1:
        raise InvalidEnsembleError(
            f"ensemble members must be one-dimensional, got {array.ndim} dimensions"
        )
    if array.size == 0:
        raise EmptyEnsembleError("ensemble members must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise InvalidEnsembleError("ensemble members must contain only finite numeric values")

    return array
