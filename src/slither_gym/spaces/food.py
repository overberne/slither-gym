from typing import Any, TypedDict, cast

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray


class Food(TypedDict):
    """
    A food or prey entity.

    Positions are expressed in the player's local coordinate frame, including
    rotation, such that the player is always at the origin facing a canonical
    direction.

    Food size is normalised to maintain values close to ``[0, 1]``.

    Attributes
    ----------
    x : NDArray[np.float32]
        X-position relative to the player slither, normalised by perception
        radius. Shape ``(1,)``.
    y : NDArray[np.float32]
        Y-position relative to the player slither, normalised by perception
        radius. Shape ``(1,)``.
    size : NDArray[np.float32]
        Normalised food size (divided by 50). Shape ``(1,)``.
    """

    x: NDArray[np.float32]
    y: NDArray[np.float32]
    size: NDArray[np.float32]


def new_food_space(**kwargs: Any) -> spaces.Space[Food]:
    """
    Construct the Gymnasium space for a food or prey entity.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments forwarded to ``spaces.Dict``.

    Returns
    -------
    spaces.Space[Food]
        A Gymnasium ``Dict`` space defining a food observation.
    """
    return cast(
        spaces.Space[Food],
        spaces.Dict(
            {
                'x': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'y': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'size': spaces.Box(0, 1, (1,), dtype=np.float32),
            },
            **kwargs,
        ),
    )
