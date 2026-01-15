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


class Action(TypedDict):
    """
    Action issued by the agent at a single decision step.

    The action specifies the desired movement direction using a sine–cosine
    representation of the heading angle, along with a binary boost control.
    Encoding the heading this way avoids angular discontinuities and provides
    a smooth action manifold for learning algorithms.

    Attributes
    ----------
    heading_cos : NDArray[np.float32]
        Cosine of the desired heading angle. Values are constrained to ``[-1, 1]``.
        Shape ``(1,)``.
    heading_sin : NDArray[np.float32]
        Sine of the desired heading angle. Values are constrained to ``[-1, 1]``.
        Shape ``(1,)``.
    boost : NDArray[np.int8]
        Whether to enable boosting during this step (binary: 0 or 1).
        Shape ``(1,)``.
    """

    heading_cos: NDArray[np.float32]
    heading_sin: NDArray[np.float32]
    boost: NDArray[np.int8]


def new_action_space(**kwargs: Any) -> spaces.Space[Action]:
    """
    Construct the Gymnasium action space for the slither environment.

    The action space consists of a desired heading direction encoded as sine
    and cosine, and a binary boost flag.

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments forwarded to ``spaces.Dict``.

    Returns
    -------
    spaces.Space[Action]
        A Gymnasium ``Dict`` space defining the action specification.
    """
    return cast(
        spaces.Space[Action],
        spaces.Dict(
            {
                'heading_cos': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'heading_sin': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'boost': spaces.MultiBinary(1),
            },
            **kwargs,
        ),
    )


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
                'player': _new_slither_space(),
                'enemies': spaces.Sequence(_new_slither_space()),
                'food': spaces.Sequence(_new_food_space()),
                'minimap': spaces.Sequence(_new_minimap_space()),
                'nearest_border_cos': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'nearest_border_sin': spaces.Box(-1, 1, (1,), dtype=np.float32),
                'distance_to_border': spaces.Box(0, 1, (1,), dtype=np.float32),
                'time_since_last_decision': spaces.Box(0, 10, (1,), dtype=np.float32),
            },
            **kwargs,
        ),
    )


def _new_segment_space(**kwargs: Any) -> spaces.Space[SlitherSegment]:
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


def _new_slither_space(**kwargs: Any) -> spaces.Space[Slither]:
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
                'segments': spaces.Sequence(_new_segment_space()),
            },
            **kwargs,
        ),
    )


def _new_food_space(**kwargs: Any) -> spaces.Space[Food]:
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


def _new_minimap_space(**kwargs: Any) -> spaces.Space[MinimapSector]:
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
