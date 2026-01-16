from typing import Any, TypedDict, cast

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray

from slither_gym.spaces.food import Food, new_food_space
from slither_gym.spaces.minimap import MinimapSector, new_minimap_space
from slither_gym.spaces.slither import Slither, new_slither_space


class Observation(TypedDict):
    """
    Full environment observation returned to the agent.

    This observation combines detailed local information (player, enemies,
    food) with coarse global context (minimap and border proximity).

    Attributes
    ----------
    player : Slither
        Observation of the player-controlled slither.
    enemies : list[Slither]
        Observations of nearby enemy slithers.
    food : list[Food]
        Observations of visible food or prey entities.
    minimap : list[MinimapSector]
        Coarse minimap representation of distant slithers.
    nearest_border_sin : NDArray[np.float32]
        Sine of the heading toward the nearest world border. Shape ``(1,)``.
    nearest_border_cos : NDArray[np.float32]
        Cosine of the heading toward the nearest world border. Shape ``(1,)``.
    distance_to_border : NDArray[np.float32]
        Distance to the nearest border, normalised by perception radius and
        capped at 1. Shape ``(1,)``.
    time_since_last_decision : NDArray[np.float32]
        Time elapsed since the last agent decision, in seconds.
        Shape ``(1,)``.
    """

    player: Slither
    enemies: list[Slither]
    food: list[Food]
    minimap: list[MinimapSector]
    nearest_border_sin: NDArray[np.float32]
    nearest_border_cos: NDArray[np.float32]
    distance_to_border: NDArray[np.float32]
    time_since_last_decision: NDArray[np.float32]


def new_observation_space(**kwargs: Any) -> spaces.Space[Observation]:
    """
    Construct the Gymnasium observation space for the slither environment.

    This space mirrors the structure of :class:`Observation` and combines
    nested ``Dict`` and ``Sequence`` spaces to support variable numbers of
    enemies, food items, and minimap sectors.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments forwarded to ``spaces.Dict``.

    Returns
    -------
    spaces.Space[Observation]
        A Gymnasium ``Dict`` space defining the observation specification.
    """
    return cast(
        spaces.Space[Observation],
        spaces.Dict(
            {
                'player': new_slither_space(),
                'enemies': spaces.Sequence(new_slither_space()),
                'food': spaces.Sequence(new_food_space()),
                'minimap': spaces.Sequence(new_minimap_space()),
                'nearest_border_cos': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'nearest_border_sin': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'distance_to_border': spaces.Box(0, 1, (1,), dtype=np.float32),
                'time_since_last_decision': spaces.Box(0, 10, (1,), dtype=np.float32),
            },
            **kwargs,
        ),
    )
