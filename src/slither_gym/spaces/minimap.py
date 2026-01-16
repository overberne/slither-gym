from typing import Any, TypedDict, cast

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray


class MinimapSector(TypedDict):
    """
    A minimap sector containing one or more slithers.

    Positions are expressed relative to the map center and normalised to
    ``[-1, 1]`` to provide a coarse global spatial context.

    Attributes
    ----------
    x : NDArray[np.float32]
        X-position of the sector center relative to the map center.
        Shape ``(1,)``.
    y : NDArray[np.float32]
        Y-position of the sector center relative to the map center.
        Shape ``(1,)``.
    """

    x: NDArray[np.float32]
    y: NDArray[np.float32]


def new_minimap_space(**kwargs: Any) -> spaces.Space[MinimapSector]:
    """
    Construct the Gymnasium space for a minimap sector.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments forwarded to ``spaces.Dict``.

    Returns
    -------
    spaces.Space[MinimapSector]
        A Gymnasium ``Dict`` space defining a minimap sector.
    """
    return cast(
        spaces.Space[MinimapSector],
        spaces.Dict(
            {
                'x': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'y': spaces.Box(-1, 1, (1,), dtype=np.float32),
            },
            **kwargs,
        ),
    )
