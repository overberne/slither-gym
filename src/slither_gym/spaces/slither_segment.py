from typing import Any, TypedDict, cast

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray


class SlitherSegment(TypedDict):
    """
    A single body segment of a slither.

    The segment position is expressed in the *local coordinate frame* of its
    parent slither rather than in absolute world coordinates. This local frame
    is rotated with the slither's heading, which improves invariance and
    generalisation in downstream encoders.

    Positions are normalised by the perception radius.

    Attributes
    ----------
    x : NDArray[np.float32]
        X-coordinate of the segment in the local slither frame, normalised by
        the perception radius. Shape ``(1,)``.
    y : NDArray[np.float32]
        Y-coordinate of the segment in the local slither frame, normalised by
        the perception radius. Shape ``(1,)``.
    """

    x: NDArray[np.float32]
    y: NDArray[np.float32]


def new_segment_space(**kwargs: Any) -> spaces.Space[SlitherSegment]:
    """
    Construct the Gymnasium space for a slither body segment.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments forwarded to ``spaces.Dict``.

    Returns
    -------
    spaces.Space[SlitherSegment]
        A Gymnasium ``Dict`` space defining a slither segment.
    """
    return cast(
        spaces.Space[SlitherSegment],
        spaces.Dict(
            {
                'x': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'y': spaces.Box(-1, 1, (1,), dtype=np.float32),
            },
            **kwargs,
        ),
    )
