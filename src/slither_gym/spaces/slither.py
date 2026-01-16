from typing import Any, TypedDict, cast

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray

from slither_gym.spaces.slither_segment import SlitherSegment, new_segment_space


class Slither(TypedDict):
    """
    Representation of a slither (player or enemy).

    Coordinate semantics depend on whether the slither represents the player
    or an enemy:

    * **Player slither**
        - ``x`` and ``y`` are relative to the center of the map.
        - Positions are normalised by the map radius.

    * **Enemy slither**
        - ``x`` and ``y`` are relative to the player slither.
        - Positions are normalised by the perception radius.

    Both global and relative motion features are included. Global measurements
    constrain feasible turning and movement dynamics, while relative
    measurements improve prediction of enemy behaviour.

    Normalisation conventions
    --------------------------
    * Speed is normalised by the maximum speed (14).
    * Length is normalised by ``SLITHER_LENGTH_NORM = 50000``.

    Attributes
    ----------
    x : NDArray[np.float32]
        X-position of the slither under the applicable coordinate convention.
        Shape ``(1,)``.
    y : NDArray[np.float32]
        Y-position of the slither under the applicable coordinate convention.
        Shape ``(1,)``.
    heading_cos : NDArray[np.float32]
        Cosine of the slither's absolute heading angle. Shape ``(1,)``.
    heading_sin : NDArray[np.float32]
        Sine of the slither's absolute heading angle. Shape ``(1,)``.
    speed : NDArray[np.float32]
        Normalised forward speed in ``[0, 1]``. Shape ``(1,)``.
    relative_heading_cos : NDArray[np.float32]
        Cosine of the heading difference between this slither and the player.
        Shape ``(1,)``.
    relative_heading_sin : NDArray[np.float32]
        Sine of the heading difference between this slither and the player.
        Shape ``(1,)``.
    relative_speed : NDArray[np.float32]
        Relative speed with respect to the player slither. Shape ``(1,)``.
    length : NDArray[np.float32]
        Normalised slither length. Shape ``(1,)``.
    boosting : NDArray[np.int8]
        Whether the slither is currently boosting (binary). Shape ``(1,)``.
    segments : list[SlitherSegment]
        Ordered list of body segments in the slither's local reference frame.
    """

    x: NDArray[np.float32]
    y: NDArray[np.float32]
    heading_cos: NDArray[np.float32]
    heading_sin: NDArray[np.float32]
    speed: NDArray[np.float32]
    relative_heading_sin: NDArray[np.float32]
    relative_heading_cos: NDArray[np.float32]
    relative_speed: NDArray[np.float32]
    length: NDArray[np.float32]
    boosting: NDArray[np.int8]
    segments: list[SlitherSegment]


def new_slither_space(**kwargs: Any) -> spaces.Space[Slither]:
    """
    Construct the Gymnasium space for a slither entity.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments forwarded to ``spaces.Dict``.

    Returns
    -------
    spaces.Space[Slither]
        A Gymnasium ``Dict`` space defining a slither observation.
    """
    return cast(
        spaces.Space[Slither],
        spaces.Dict(
            {
                'x': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'y': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'heading_cos': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'heading_sin': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'speed': spaces.Box(0.0, 1, (1,), dtype=np.float32),
                'relative_heading_cos': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'relative_heading_sin': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'relative_speed': spaces.Box(-2, 2, (1,), dtype=np.float32),
                'length': spaces.Box(0, 1, (1,), dtype=np.float32),
                'boosting': spaces.MultiBinary(1),
                'segments': spaces.Sequence(new_segment_space()),
            },
            **kwargs,
        ),
    )
