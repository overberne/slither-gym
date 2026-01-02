from typing import TypedDict, cast

import gymnasium.spaces as spaces
import gymnasium.vector as gym
import numpy as np
from numpy.typing import NDArray

from slither_gym import Action


class VectorAction(TypedDict):
    heading: NDArray[np.float32]
    boost: NDArray[np.int32]


class UseTrigonometricVectorAction[ObsType, ActType, ArrayType](gym.VectorActionWrapper):
    def __init__(self, env: gym.VectorEnv[ObsType, ActType, ArrayType]):
        super().__init__(env)  # type: ignore
        self.single_action_space = cast(
            spaces.Space[Action],
            spaces.Dict(
                {
                    'heading': spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                    'boost': spaces.Discrete(2, dtype=np.int32),
                }
            ),
        )
        self.action_space = gym.utils.batch_space(self.single_action_space, env.num_envs)

    def actions(self, actions: VectorAction) -> VectorAction:  # type: ignore
        # action['heading'].shape == (num_envs, 2)
        sin = actions['heading'][:, 0]
        cos = actions['heading'][:, 1]

        actions = actions.copy()
        actions['heading'] = np.atan2(sin, cos)
        return actions
