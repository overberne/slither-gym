from typing import Any, TypedDict, cast

import gymnasium.spaces as spaces
import numpy as np
from numpy.typing import NDArray


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
